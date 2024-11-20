def main():
    args = parse_args()
    if args.report_to == 'wandb' and args.hub_token is not None:
        raise ValueError(
            'You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token. Please use `huggingface-cli login` to authenticate with the Hub.'
            )
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
    if jax.process_index() == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.
                output_dir).name, exist_ok=True, token=args.hub_token).repo_id
    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name, args.dataset_config_name,
            cache_dir=args.cache_dir, data_dir=args.train_data_dir)
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files['train'] = os.path.join(args.train_data_dir, '**')
        dataset = load_dataset('imagefolder', data_files=data_files,
            cache_dir=args.cache_dir)
    column_names = dataset['train'].column_names
    dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0
            ] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
                )
    if args.caption_column is None:
        caption_column = dataset_columns[1
            ] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
                )

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else
                    caption[0])
            else:
                raise ValueError(
                    f'Caption column `{caption_column}` should contain either strings or lists of strings.'
                    )
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length,
            padding='do_not_pad', truncation=True)
        input_ids = inputs.input_ids
        return input_ids
    train_transforms = transforms.Compose([transforms.Resize(args.
        resolution, interpolation=transforms.InterpolationMode.BILINEAR), 
        transforms.CenterCrop(args.resolution) if args.center_crop else
        transforms.RandomCrop(args.resolution), transforms.
        RandomHorizontalFlip() if args.random_flip else transforms.Lambda(
        lambda x: x), transforms.ToTensor(), transforms.Normalize([0.5], [
        0.5])])

    def preprocess_train(examples):
        images = [image.convert('RGB') for image in examples[image_column]]
        examples['pixel_values'] = [train_transforms(image) for image in images
            ]
        examples['input_ids'] = tokenize_captions(examples)
        return examples
    if args.max_train_samples is not None:
        dataset['train'] = dataset['train'].shuffle(seed=args.seed).select(
            range(args.max_train_samples))
    train_dataset = dataset['train'].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example['pixel_values'] for example in
            examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format
            ).float()
        input_ids = [example['input_ids'] for example in examples]
        padded_tokens = tokenizer.pad({'input_ids': input_ids}, padding=
            'max_length', max_length=tokenizer.model_max_length,
            return_tensors='pt')
        batch = {'pixel_values': pixel_values, 'input_ids': padded_tokens.
            input_ids}
        batch = {k: v.numpy() for k, v in batch.items()}
        return batch
    total_train_batch_size = args.train_batch_size * jax.local_device_count()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=
        True, collate_fn=collate_fn, batch_size=total_train_batch_size,
        drop_last=True)
    weight_dtype = jnp.float32
    if args.mixed_precision == 'fp16':
        weight_dtype = jnp.float16
    elif args.mixed_precision == 'bf16':
        weight_dtype = jnp.bfloat16
    tokenizer = CLIPTokenizer.from_pretrained(args.
        pretrained_model_name_or_path, from_pt=args.from_pt, revision=args.
        revision, subfolder='tokenizer')
    text_encoder = FlaxCLIPTextModel.from_pretrained(args.
        pretrained_model_name_or_path, from_pt=args.from_pt, revision=args.
        revision, subfolder='text_encoder', dtype=weight_dtype)
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(args.
        pretrained_model_name_or_path, from_pt=args.from_pt, revision=args.
        revision, subfolder='vae', dtype=weight_dtype)
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(args.
        pretrained_model_name_or_path, from_pt=args.from_pt, revision=args.
        revision, subfolder='unet', dtype=weight_dtype)
    if args.scale_lr:
        args.learning_rate = args.learning_rate * total_train_batch_size
    constant_scheduler = optax.constant_schedule(args.learning_rate)
    adamw = optax.adamw(learning_rate=constant_scheduler, b1=args.
        adam_beta1, b2=args.adam_beta2, eps=args.adam_epsilon, weight_decay
        =args.adam_weight_decay)
    optimizer = optax.chain(optax.clip_by_global_norm(args.max_grad_norm),
        adamw)
    state = train_state.TrainState.create(apply_fn=unet.__call__, params=
        unet_params, tx=optimizer)
    noise_scheduler = FlaxDDPMScheduler(beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', num_train_timesteps=1000)
    noise_scheduler_state = noise_scheduler.create_state()
    rng = jax.random.PRNGKey(args.seed)
    train_rngs = jax.random.split(rng, jax.local_device_count())

    def train_step(state, text_encoder_params, vae_params, batch, train_rng):
        dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)

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
            encoder_hidden_states = text_encoder(batch['input_ids'], params
                =text_encoder_params, train=False)[0]
            model_pred = unet.apply({'params': params}, noisy_latents,
                timesteps, encoder_hidden_states, train=True).sample
            if noise_scheduler.config.prediction_type == 'epsilon':
                target = noise
            elif noise_scheduler.config.prediction_type == 'v_prediction':
                target = noise_scheduler.get_velocity(noise_scheduler_state,
                    latents, noise, timesteps)
            else:
                raise ValueError(
                    f'Unknown prediction type {noise_scheduler.config.prediction_type}'
                    )
            loss = (target - model_pred) ** 2
            loss = loss.mean()
            return loss
        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, 'batch')
        new_state = state.apply_gradients(grads=grad)
        metrics = {'loss': loss}
        metrics = jax.lax.pmean(metrics, axis_name='batch')
        return new_state, metrics, new_train_rng
    p_train_step = jax.pmap(train_step, 'batch', donate_argnums=(0,))
    state = jax_utils.replicate(state)
    text_encoder_params = jax_utils.replicate(text_encoder.params)
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
    global_step = 0
    epochs = tqdm(range(args.num_train_epochs), desc='Epoch ... ', position=0)
    for epoch in epochs:
        train_metrics = []
        steps_per_epoch = len(train_dataset) // total_train_batch_size
        train_step_progress_bar = tqdm(total=steps_per_epoch, desc=
            'Training...', position=1, leave=False)
        for batch in train_dataloader:
            batch = shard(batch)
            state, train_metric, train_rngs = p_train_step(state,
                text_encoder_params, vae_params, batch, train_rngs)
            train_metrics.append(train_metric)
            train_step_progress_bar.update(1)
            global_step += 1
            if global_step >= args.max_train_steps:
                break
        train_metric = jax_utils.unreplicate(train_metric)
        train_step_progress_bar.close()
        epochs.write(
            f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss: {train_metric['loss']})"
            )
    if jax.process_index() == 0:
        scheduler = FlaxPNDMScheduler(beta_start=0.00085, beta_end=0.012,
            beta_schedule='scaled_linear', skip_prk_steps=True)
        safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
            'CompVis/stable-diffusion-safety-checker', from_pt=True)
        pipeline = FlaxStableDiffusionPipeline(text_encoder=text_encoder,
            vae=vae, unet=unet, tokenizer=tokenizer, scheduler=scheduler,
            safety_checker=safety_checker, feature_extractor=
            CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32'))
        pipeline.save_pretrained(args.output_dir, params={'text_encoder':
            get_params_to_save(text_encoder_params), 'vae':
            get_params_to_save(vae_params), 'unet': get_params_to_save(
            state.params), 'safety_checker': safety_checker.params})
        if args.push_to_hub:
            upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message='End of training', ignore_patterns=['step_*',
                'epoch_*'])
