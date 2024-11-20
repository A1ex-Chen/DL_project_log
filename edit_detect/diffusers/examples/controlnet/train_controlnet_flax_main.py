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
    if jax.process_index() == 0 and args.report_to == 'wandb':
        wandb.init(entity=args.wandb_entity, project=args.
            tracker_project_name, job_type='train', config=args)
    if args.seed is not None:
        set_seed(args.seed)
    rng = jax.random.PRNGKey(0)
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
    total_train_batch_size = args.train_batch_size * jax.local_device_count(
        ) * args.gradient_accumulation_steps
    train_dataset = make_train_dataset(args, tokenizer, batch_size=
        total_train_batch_size)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=
        not args.streaming, collate_fn=collate_fn, batch_size=
        total_train_batch_size, num_workers=args.dataloader_num_workers,
        drop_last=True)
    weight_dtype = jnp.float32
    if args.mixed_precision == 'fp16':
        weight_dtype = jnp.float16
    elif args.mixed_precision == 'bf16':
        weight_dtype = jnp.bfloat16
    text_encoder = FlaxCLIPTextModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='text_encoder', dtype=
        weight_dtype, revision=args.revision, from_pt=args.from_pt)
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(args.
        pretrained_model_name_or_path, revision=args.revision, subfolder=
        'vae', dtype=weight_dtype, from_pt=args.from_pt)
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='unet', dtype=weight_dtype,
        revision=args.revision, from_pt=args.from_pt)
    if args.controlnet_model_name_or_path:
        logger.info('Loading existing controlnet weights')
        controlnet, controlnet_params = FlaxControlNetModel.from_pretrained(
            args.controlnet_model_name_or_path, revision=args.
            controlnet_revision, from_pt=args.controlnet_from_pt, dtype=jnp
            .float32)
    else:
        logger.info('Initializing controlnet weights from unet')
        rng, rng_params = jax.random.split(rng)
        controlnet = FlaxControlNetModel(in_channels=unet.config.
            in_channels, down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            attention_head_dim=unet.config.attention_head_dim,
            cross_attention_dim=unet.config.cross_attention_dim,
            use_linear_projection=unet.config.use_linear_projection,
            flip_sin_to_cos=unet.config.flip_sin_to_cos, freq_shift=unet.
            config.freq_shift)
        controlnet_params = controlnet.init_weights(rng=rng_params)
        controlnet_params = unfreeze(controlnet_params)
        for key in ['conv_in', 'time_embedding', 'down_blocks_0',
            'down_blocks_1', 'down_blocks_2', 'down_blocks_3', 'mid_block']:
            controlnet_params[key] = unet_params[key]
    pipeline, pipeline_params = (FlaxStableDiffusionControlNetPipeline.
        from_pretrained(args.pretrained_model_name_or_path, tokenizer=
        tokenizer, controlnet=controlnet, safety_checker=None, dtype=
        weight_dtype, revision=args.revision, from_pt=args.from_pt))
    pipeline_params = jax_utils.replicate(pipeline_params)
    if args.scale_lr:
        args.learning_rate = args.learning_rate * total_train_batch_size
    constant_scheduler = optax.constant_schedule(args.learning_rate)
    adamw = optax.adamw(learning_rate=constant_scheduler, b1=args.
        adam_beta1, b2=args.adam_beta2, eps=args.adam_epsilon, weight_decay
        =args.adam_weight_decay)
    optimizer = optax.chain(optax.clip_by_global_norm(args.max_grad_norm),
        adamw)
    state = train_state.TrainState.create(apply_fn=controlnet.__call__,
        params=controlnet_params, tx=optimizer)
    noise_scheduler, noise_scheduler_state = FlaxDDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder='scheduler')
    validation_rng, train_rngs = jax.random.split(rng)
    train_rngs = jax.random.split(train_rngs, jax.local_device_count())

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler_state.common.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
        alpha = sqrt_alphas_cumprod[timesteps]
        sigma = sqrt_one_minus_alphas_cumprod[timesteps]
        snr = (alpha / sigma) ** 2
        return snr

    def train_step(state, unet_params, text_encoder_params, vae_params,
        batch, train_rng):
        if args.gradient_accumulation_steps > 1:
            grad_steps = args.gradient_accumulation_steps
            batch = jax.tree_map(lambda x: x.reshape((grad_steps, x.shape[0
                ] // grad_steps) + x.shape[1:]), batch)

        def compute_loss(params, minibatch, sample_rng):
            vae_outputs = vae.apply({'params': vae_params}, minibatch[
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
            encoder_hidden_states = text_encoder(minibatch['input_ids'],
                params=text_encoder_params, train=False)[0]
            controlnet_cond = minibatch['conditioning_pixel_values']
            down_block_res_samples, mid_block_res_sample = controlnet.apply({
                'params': params}, noisy_latents, timesteps,
                encoder_hidden_states, controlnet_cond, train=True,
                return_dict=False)
            model_pred = unet.apply({'params': unet_params}, noisy_latents,
                timesteps, encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample).sample
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
            if args.snr_gamma is not None:
                snr = jnp.array(compute_snr(timesteps))
                snr_loss_weights = jnp.where(snr < args.snr_gamma, snr, jnp
                    .ones_like(snr) * args.snr_gamma)
                if noise_scheduler.config.prediction_type == 'epsilon':
                    snr_loss_weights = snr_loss_weights / snr
                elif noise_scheduler.config.prediction_type == 'v_prediction':
                    snr_loss_weights = snr_loss_weights / (snr + 1)
                loss = loss * snr_loss_weights
            loss = loss.mean()
            return loss
        grad_fn = jax.value_and_grad(compute_loss)

        def get_minibatch(batch, grad_idx):
            return jax.tree_util.tree_map(lambda x: jax.lax.
                dynamic_index_in_dim(x, grad_idx, keepdims=False), batch)

        def loss_and_grad(grad_idx, train_rng):
            minibatch = get_minibatch(batch, grad_idx
                ) if grad_idx is not None else batch
            sample_rng, train_rng = jax.random.split(train_rng, 2)
            loss, grad = grad_fn(state.params, minibatch, sample_rng)
            return loss, grad, train_rng
        if args.gradient_accumulation_steps == 1:
            loss, grad, new_train_rng = loss_and_grad(None, train_rng)
        else:
            init_loss_grad_rng = 0.0, jax.tree_map(jnp.zeros_like, state.params
                ), train_rng

            def cumul_grad_step(grad_idx, loss_grad_rng):
                cumul_loss, cumul_grad, train_rng = loss_grad_rng
                loss, grad, new_train_rng = loss_and_grad(grad_idx, train_rng)
                cumul_loss, cumul_grad = jax.tree_map(jnp.add, (cumul_loss,
                    cumul_grad), (loss, grad))
                return cumul_loss, cumul_grad, new_train_rng
            loss, grad, new_train_rng = jax.lax.fori_loop(0, args.
                gradient_accumulation_steps, cumul_grad_step,
                init_loss_grad_rng)
            loss, grad = jax.tree_map(lambda x: x / args.
                gradient_accumulation_steps, (loss, grad))
        grad = jax.lax.pmean(grad, 'batch')
        new_state = state.apply_gradients(grads=grad)
        metrics = {'loss': loss}
        metrics = jax.lax.pmean(metrics, axis_name='batch')

        def l2(xs):
            return jnp.sqrt(sum([jnp.vdot(x, x) for x in jax.tree_util.
                tree_leaves(xs)]))
        metrics['l2_grads'] = l2(jax.tree_util.tree_leaves(grad))
        return new_state, metrics, new_train_rng
    p_train_step = jax.pmap(train_step, 'batch', donate_argnums=(0,))
    state = jax_utils.replicate(state)
    unet_params = jax_utils.replicate(unet_params)
    text_encoder_params = jax_utils.replicate(text_encoder.params)
    vae_params = jax_utils.replicate(vae_params)
    if args.streaming:
        dataset_length = args.max_train_samples
    else:
        dataset_length = len(train_dataloader)
    num_update_steps_per_epoch = math.ceil(dataset_length / args.
        gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
    args.num_train_epochs = math.ceil(args.max_train_steps /
        num_update_steps_per_epoch)
    logger.info('***** Running training *****')
    logger.info(
        f'  Num examples = {args.max_train_samples if args.streaming else len(train_dataset)}'
        )
    logger.info(f'  Num Epochs = {args.num_train_epochs}')
    logger.info(
        f'  Instantaneous batch size per device = {args.train_batch_size}')
    logger.info(
        f'  Total train batch size (w. parallel & distributed) = {total_train_batch_size}'
        )
    logger.info(
        f'  Total optimization steps = {args.num_train_epochs * num_update_steps_per_epoch}'
        )
    if jax.process_index() == 0 and args.report_to == 'wandb':
        wandb.define_metric('*', step_metric='train/step')
        wandb.define_metric('train/step', step_metric='walltime')
        wandb.config.update({'num_train_examples': args.max_train_samples if
            args.streaming else len(train_dataset),
            'total_train_batch_size': total_train_batch_size,
            'total_optimization_step': args.num_train_epochs *
            num_update_steps_per_epoch, 'num_devices': jax.device_count(),
            'controlnet_params': sum(np.prod(x.shape) for x in jax.
            tree_util.tree_leaves(state.params))})
    global_step = step0 = 0
    epochs = tqdm(range(args.num_train_epochs), desc='Epoch ... ', position
        =0, disable=jax.process_index() > 0)
    if args.profile_memory:
        jax.profiler.save_device_memory_profile(os.path.join(args.
            output_dir, 'memory_initial.prof'))
    t00 = t0 = time.monotonic()
    for epoch in epochs:
        train_metrics = []
        train_metric = None
        steps_per_epoch = (args.max_train_samples // total_train_batch_size if
            args.streaming or args.max_train_samples else len(train_dataset
            ) // total_train_batch_size)
        train_step_progress_bar = tqdm(total=steps_per_epoch, desc=
            'Training...', position=1, leave=False, disable=jax.
            process_index() > 0)
        for batch in train_dataloader:
            if args.profile_steps and global_step == 1:
                train_metric['loss'].block_until_ready()
                jax.profiler.start_trace(args.output_dir)
            if args.profile_steps and global_step == 1 + args.profile_steps:
                train_metric['loss'].block_until_ready()
                jax.profiler.stop_trace()
            batch = shard(batch)
            with jax.profiler.StepTraceAnnotation('train', step_num=global_step
                ):
                state, train_metric, train_rngs = p_train_step(state,
                    unet_params, text_encoder_params, vae_params, batch,
                    train_rngs)
            train_metrics.append(train_metric)
            train_step_progress_bar.update(1)
            global_step += 1
            if global_step >= args.max_train_steps:
                break
            if (args.validation_prompt is not None and global_step % args.
                validation_steps == 0 and jax.process_index() == 0):
                _ = log_validation(pipeline, pipeline_params, state.params,
                    tokenizer, args, validation_rng, weight_dtype)
            if global_step % args.logging_steps == 0 and jax.process_index(
                ) == 0:
                if args.report_to == 'wandb':
                    train_metrics = jax_utils.unreplicate(train_metrics)
                    train_metrics = jax.tree_util.tree_map(lambda *m: jnp.
                        array(m).mean(), *train_metrics)
                    wandb.log({'walltime': time.monotonic() - t00,
                        'train/step': global_step, 'train/epoch': 
                        global_step / dataset_length, 'train/steps_per_sec':
                        (global_step - step0) / (time.monotonic() - t0), **
                        {f'train/{k}': v for k, v in train_metrics.items()}})
                t0, step0 = time.monotonic(), global_step
                train_metrics = []
            if (global_step % args.checkpointing_steps == 0 and jax.
                process_index() == 0):
                controlnet.save_pretrained(f'{args.output_dir}/{global_step}',
                    params=get_params_to_save(state.params))
        train_metric = jax_utils.unreplicate(train_metric)
        train_step_progress_bar.close()
        epochs.write(
            f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss: {train_metric['loss']})"
            )
    if jax.process_index() == 0:
        if args.validation_prompt is not None:
            if args.profile_validation:
                jax.profiler.start_trace(args.output_dir)
            image_logs = log_validation(pipeline, pipeline_params, state.
                params, tokenizer, args, validation_rng, weight_dtype)
            if args.profile_validation:
                jax.profiler.stop_trace()
        else:
            image_logs = None
        controlnet.save_pretrained(args.output_dir, params=
            get_params_to_save(state.params))
        if args.push_to_hub:
            save_model_card(repo_id, image_logs=image_logs, base_model=args
                .pretrained_model_name_or_path, repo_folder=args.output_dir)
            upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message='End of training', ignore_patterns=['step_*',
                'epoch_*'])
    if args.profile_memory:
        jax.profiler.save_device_memory_profile(os.path.join(args.
            output_dir, 'memory_final.prof'))
    logger.info('Finished training.')
