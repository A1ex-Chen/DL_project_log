def main():
    args = parse_args()
    logging.basicConfig(format=
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt=
        '%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR
        )
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))
        if cur_class_images < args.num_class_images:
            pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(args
                .pretrained_model_name_or_path, safety_checker=None,
                revision=args.revision)
            pipeline.set_progress_bar_config(disable=True)
            num_new_images = args.num_class_images - cur_class_images
            logger.info(f'Number of class images to sample: {num_new_images}.')
            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            total_sample_batch_size = (args.sample_batch_size * jax.
                local_device_count())
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset,
                batch_size=total_sample_batch_size)
            for example in tqdm(sample_dataloader, desc=
                'Generating class images', disable=not jax.process_index() == 0
                ):
                prompt_ids = pipeline.prepare_inputs(example['prompt'])
                prompt_ids = shard(prompt_ids)
                p_params = jax_utils.replicate(params)
                rng = jax.random.split(rng)[0]
                sample_rng = jax.random.split(rng, jax.device_count())
                images = pipeline(prompt_ids, p_params, sample_rng, jit=True
                    ).images
                images = images.reshape((images.shape[0] * images.shape[1],
                    ) + images.shape[-3:])
                images = pipeline.numpy_to_pil(np.array(images))
                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()
                        ).hexdigest()
                    image_filename = (class_images_dir /
                        f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        )
                    image.save(image_filename)
            del pipeline
    if jax.process_index() == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.
                output_dir).name, exist_ok=True, token=args.hub_token).repo_id
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.
            pretrained_model_name_or_path, subfolder='tokenizer', revision=
            args.revision)
    else:
        raise NotImplementedError('No tokenizer specified!')
    train_dataset = DreamBoothDataset(instance_data_root=args.
        instance_data_dir, instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation
         else None, class_prompt=args.class_prompt, class_num=args.
        num_class_images, tokenizer=tokenizer, size=args.resolution,
        center_crop=args.center_crop)

    def collate_fn(examples):
        input_ids = [example['instance_prompt_ids'] for example in examples]
        pixel_values = [example['instance_images'] for example in examples]
        if args.with_prior_preservation:
            input_ids += [example['class_prompt_ids'] for example in examples]
            pixel_values += [example['class_images'] for example in examples]
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format
            ).float()
        input_ids = tokenizer.pad({'input_ids': input_ids}, padding=
            'max_length', max_length=tokenizer.model_max_length,
            return_tensors='pt').input_ids
        batch = {'input_ids': input_ids, 'pixel_values': pixel_values}
        batch = {k: v.numpy() for k, v in batch.items()}
        return batch
    total_train_batch_size = args.train_batch_size * jax.local_device_count()
    if len(train_dataset) < total_train_batch_size:
        raise ValueError(
            f"Training batch size is {total_train_batch_size}, but your dataset only contains {len(train_dataset)} images. Please, use a larger dataset or reduce the effective batch size. Note that there are {jax.local_device_count()} parallel devices, so your batch size can't be smaller than that."
            )
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=total_train_batch_size, shuffle=True, collate_fn=
        collate_fn, drop_last=True)
    weight_dtype = jnp.float32
    if args.mixed_precision == 'fp16':
        weight_dtype = jnp.float16
    elif args.mixed_precision == 'bf16':
        weight_dtype = jnp.bfloat16
    if args.pretrained_vae_name_or_path:
        vae_arg, vae_kwargs = args.pretrained_vae_name_or_path, {'from_pt':
            True}
    else:
        vae_arg, vae_kwargs = args.pretrained_model_name_or_path, {'subfolder':
            'vae', 'revision': args.revision}
    text_encoder = FlaxCLIPTextModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='text_encoder', dtype=
        weight_dtype, revision=args.revision)
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(vae_arg, dtype=
        weight_dtype, **vae_kwargs)
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='unet', dtype=weight_dtype,
        revision=args.revision)
    if args.scale_lr:
        args.learning_rate = args.learning_rate * total_train_batch_size
    constant_scheduler = optax.constant_schedule(args.learning_rate)
    adamw = optax.adamw(learning_rate=constant_scheduler, b1=args.
        adam_beta1, b2=args.adam_beta2, eps=args.adam_epsilon, weight_decay
        =args.adam_weight_decay)
    optimizer = optax.chain(optax.clip_by_global_norm(args.max_grad_norm),
        adamw)
    unet_state = train_state.TrainState.create(apply_fn=unet.__call__,
        params=unet_params, tx=optimizer)
    text_encoder_state = train_state.TrainState.create(apply_fn=
        text_encoder.__call__, params=text_encoder.params, tx=optimizer)
    noise_scheduler = FlaxDDPMScheduler(beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', num_train_timesteps=1000)
    noise_scheduler_state = noise_scheduler.create_state()
    train_rngs = jax.random.split(rng, jax.local_device_count())

    def train_step(unet_state, text_encoder_state, vae_params, batch, train_rng
        ):
        dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)
        if args.train_text_encoder:
            params = {'text_encoder': text_encoder_state.params, 'unet':
                unet_state.params}
        else:
            params = {'unet': unet_state.params}

        def compute_loss(params):
            vae_outputs = vae.apply({'params': vae_params}, batch[
                'pixel_values'], deterministic=True, method=vae.encode)
            latents = vae_outputs.latent_dist.sample(sample_rng)
            latents = jnp.transpose(latents, (0, 3, 1, 2))
            latents = latents * vae.config.scaling_factor
            noise_rng, timestep_rng = jax.random.split(sample_rng)
            noise = jax.random.normal(noise_rng, latents.shape)
            bsz = latents.shape[0]
            timesteps = jax.random.randint(timestep_rng, (bsz,), 0,
                noise_scheduler.config.num_train_timesteps)
            noisy_latents = noise_scheduler.add_noise(noise_scheduler_state,
                latents, noise, timesteps)
            if args.train_text_encoder:
                encoder_hidden_states = text_encoder_state.apply_fn(batch[
                    'input_ids'], params=params['text_encoder'],
                    dropout_rng=dropout_rng, train=True)[0]
            else:
                encoder_hidden_states = text_encoder(batch['input_ids'],
                    params=text_encoder_state.params, train=False)[0]
            model_pred = unet.apply({'params': params['unet']},
                noisy_latents, timesteps, encoder_hidden_states, train=True
                ).sample
            if noise_scheduler.config.prediction_type == 'epsilon':
                target = noise
            elif noise_scheduler.config.prediction_type == 'v_prediction':
                target = noise_scheduler.get_velocity(noise_scheduler_state,
                    latents, noise, timesteps)
            else:
                raise ValueError(
                    f'Unknown prediction type {noise_scheduler.config.prediction_type}'
                    )
            if args.with_prior_preservation:
                model_pred, model_pred_prior = jnp.split(model_pred, 2, axis=0)
                target, target_prior = jnp.split(target, 2, axis=0)
                loss = (target - model_pred) ** 2
                loss = loss.mean()
                prior_loss = (target_prior - model_pred_prior) ** 2
                prior_loss = prior_loss.mean()
                loss = loss + args.prior_loss_weight * prior_loss
            else:
                loss = (target - model_pred) ** 2
                loss = loss.mean()
            return loss
        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(params)
        grad = jax.lax.pmean(grad, 'batch')
        new_unet_state = unet_state.apply_gradients(grads=grad['unet'])
        if args.train_text_encoder:
            new_text_encoder_state = text_encoder_state.apply_gradients(grads
                =grad['text_encoder'])
        else:
            new_text_encoder_state = text_encoder_state
        metrics = {'loss': loss}
        metrics = jax.lax.pmean(metrics, axis_name='batch')
        return new_unet_state, new_text_encoder_state, metrics, new_train_rng
    p_train_step = jax.pmap(train_step, 'batch', donate_argnums=(0, 1))
    unet_state = jax_utils.replicate(unet_state)
    text_encoder_state = jax_utils.replicate(text_encoder_state)
    vae_params = jax_utils.replicate(vae_params)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    if args.max_train_steps is None:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
    args.num_train_epochs = math.ceil(args.max_train_steps /
        num_update_steps_per_epoch)
    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f'  Num Epochs = {args.num_train_epochs}')
    logger.info(
        f'  Instantaneous batch size per device = {args.train_batch_size}')
    logger.info(
        f'  Total train batch size (w. parallel & distributed) = {total_train_batch_size}'
        )
    logger.info(f'  Total optimization steps = {args.max_train_steps}')

    def checkpoint(step=None):
        scheduler, _ = FlaxPNDMScheduler.from_pretrained(
            'CompVis/stable-diffusion-v1-4', subfolder='scheduler')
        safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
            'CompVis/stable-diffusion-safety-checker', from_pt=True)
        pipeline = FlaxStableDiffusionPipeline(text_encoder=text_encoder,
            vae=vae, unet=unet, tokenizer=tokenizer, scheduler=scheduler,
            safety_checker=safety_checker, feature_extractor=
            CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32'))
        outdir = os.path.join(args.output_dir, str(step)
            ) if step else args.output_dir
        pipeline.save_pretrained(outdir, params={'text_encoder':
            get_params_to_save(text_encoder_state.params), 'vae':
            get_params_to_save(vae_params), 'unet': get_params_to_save(
            unet_state.params), 'safety_checker': safety_checker.params})
        if args.push_to_hub:
            message = (f'checkpoint-{step}' if step is not None else
                'End of training')
            upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message=message, ignore_patterns=['step_*', 'epoch_*'])
    global_step = 0
    epochs = tqdm(range(args.num_train_epochs), desc='Epoch ... ', position=0)
    for epoch in epochs:
        train_metrics = []
        steps_per_epoch = len(train_dataset) // total_train_batch_size
        train_step_progress_bar = tqdm(total=steps_per_epoch, desc=
            'Training...', position=1, leave=False)
        for batch in train_dataloader:
            batch = shard(batch)
            unet_state, text_encoder_state, train_metric, train_rngs = (
                p_train_step(unet_state, text_encoder_state, vae_params,
                batch, train_rngs))
            train_metrics.append(train_metric)
            train_step_progress_bar.update(jax.local_device_count())
            global_step += 1
            if jax.process_index(
                ) == 0 and args.save_steps and global_step % args.save_steps == 0:
                checkpoint(global_step)
            if global_step >= args.max_train_steps:
                break
        train_metric = jax_utils.unreplicate(train_metric)
        train_step_progress_bar.close()
        epochs.write(
            f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss: {train_metric['loss']})"
            )
    if jax.process_index() == 0:
        checkpoint()
