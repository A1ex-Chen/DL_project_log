def main(args):
    if args.report_to == 'wandb' and args.hub_token is not None:
        raise ValueError(
            'You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token. Please use `huggingface-cli login` to authenticate with the Hub.'
            )
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.
        output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(gradient_accumulation_steps=args.
        gradient_accumulation_steps, mixed_precision=args.mixed_precision,
        log_with=args.report_to, project_config=accelerator_project_config)
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    logging.basicConfig(format=
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt=
        '%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.
                output_dir).name, exist_ok=True, token=args.hub_token,
                private=True).repo_id
    tokenizer_one = AutoTokenizer.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='tokenizer', revision=args
        .revision, use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='tokenizer_2', revision=
        args.revision, use_fast=False)
    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.
        pretrained_model_name_or_path, args.revision)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(args.
        pretrained_model_name_or_path, args.revision, subfolder=
        'text_encoder_2')
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='scheduler')
    text_encoder_one = text_encoder_cls_one.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='text_encoder', revision=
        args.revision, variant=args.variant)
    text_encoder_two = text_encoder_cls_two.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='text_encoder_2', revision
        =args.revision, variant=args.variant)
    vae_path = (args.pretrained_model_name_or_path if args.
        pretrained_vae_model_name_or_path is None else args.
        pretrained_vae_model_name_or_path)
    vae = AutoencoderKL.from_pretrained(vae_path, subfolder='vae' if args.
        pretrained_vae_model_name_or_path is None else None, revision=args.
        revision, variant=args.variant)
    unet = UNet2DConditionModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='unet', revision=args.
        revision, variant=args.variant)
    if args.adapter_model_name_or_path:
        logger.info('Loading existing adapter weights.')
        t2iadapter = T2IAdapter.from_pretrained(args.adapter_model_name_or_path
            )
    else:
        logger.info('Initializing t2iadapter weights.')
        t2iadapter = T2IAdapter(in_channels=3, channels=(320, 640, 1280, 
            1280), num_res_blocks=2, downscale_factor=16, adapter_type=
            'full_adapter_xl')
    if version.parse(accelerate.__version__) >= version.parse('0.16.0'):

        def save_model_hook(models, weights, output_dir):
            i = len(weights) - 1
            while len(weights) > 0:
                weights.pop()
                model = models[i]
                sub_dir = 't2iadapter'
                model.save_pretrained(os.path.join(output_dir, sub_dir))
                i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                load_model = T2IAdapter.from_pretrained(os.path.join(
                    input_dir, 't2iadapter'))
                if args.control_type != 'style':
                    model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    t2iadapter.train()
    unet.train()
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse('0.0.16'):
                logger.warning(
                    'xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details.'
                    )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                'xformers is not available. Make sure it is installed correctly'
                )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    low_precision_error_string = (
        ' Please make sure to always have all model weights in full float32 precision when starting training - even if doing mixed precision training, copy of the weights should still be float32.'
        )
    if unwrap_model(t2iadapter).dtype != torch.float32:
        raise ValueError(
            f'Controlnet loaded as datatype {unwrap_model(t2iadapter).dtype}. {low_precision_error_string}'
            )
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.
            gradient_accumulation_steps * args.train_batch_size *
            accelerator.num_processes)
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                'To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.'
                )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    params_to_optimize = t2iadapter.parameters()
    optimizer = optimizer_class(params_to_optimize, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.
        adam_weight_decay, eps=args.adam_epsilon)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    else:
        vae.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    def compute_embeddings(batch, proportion_empty_prompts, text_encoders,
        tokenizers, is_train=True):
        original_size = args.resolution, args.resolution
        target_size = args.resolution, args.resolution
        crops_coords_top_left = (args.crops_coords_top_left_h, args.
            crops_coords_top_left_w)
        prompt_batch = batch[args.caption_column]
        prompt_embeds, pooled_prompt_embeds = encode_prompt(prompt_batch,
            text_encoders, tokenizers, proportion_empty_prompts, is_train)
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = list(original_size + crops_coords_top_left + target_size
            )
        add_time_ids = torch.tensor([add_time_ids])
        prompt_embeds = prompt_embeds.to(accelerator.device)
        add_text_embeds = add_text_embeds.to(accelerator.device)
        add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
        add_time_ids = add_time_ids.to(accelerator.device, dtype=
            prompt_embeds.dtype)
        unet_added_cond_kwargs = {'text_embeds': add_text_embeds,
            'time_ids': add_time_ids}
        return {'prompt_embeds': prompt_embeds, **unet_added_cond_kwargs}

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype
            =dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in
            timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]
    train_dataset = get_train_dataset(args, accelerator)
    compute_embeddings_fn = functools.partial(compute_embeddings,
        proportion_empty_prompts=args.proportion_empty_prompts,
        text_encoders=text_encoders, tokenizers=tokenizers)
    with accelerator.main_process_first():
        from datasets.fingerprint import Hasher
        new_fingerprint = Hasher.hash(args)
        train_dataset = train_dataset.map(compute_embeddings_fn, batched=
            True, new_fingerprint=new_fingerprint)
    train_dataset = prepare_train_dataset(train_dataset, accelerator)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=
        True, collate_fn=collate_fn, batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers)
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps, num_training_steps=args.
        max_train_steps, num_cycles=args.lr_num_cycles, power=args.lr_power)
    t2iadapter, optimizer, train_dataloader, lr_scheduler = (accelerator.
        prepare(t2iadapter, optimizer, train_dataloader, lr_scheduler))
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
    args.num_train_epochs = math.ceil(args.max_train_steps /
        num_update_steps_per_epoch)
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop('validation_prompt')
        tracker_config.pop('validation_image')
        accelerator.init_trackers(args.tracker_project_name, config=
            tracker_config)
    total_batch_size = (args.train_batch_size * accelerator.num_processes *
        args.gradient_accumulation_steps)
    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f'  Num batches each epoch = {len(train_dataloader)}')
    logger.info(f'  Num Epochs = {args.num_train_epochs}')
    logger.info(
        f'  Instantaneous batch size per device = {args.train_batch_size}')
    logger.info(
        f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}'
        )
    logger.info(
        f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}')
    logger.info(f'  Total optimization steps = {args.max_train_steps}')
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != 'latest':
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith('checkpoint')]
            dirs = sorted(dirs, key=lambda x: int(x.split('-')[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f'Resuming from checkpoint {path}')
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split('-')[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0
    progress_bar = tqdm(range(0, args.max_train_steps), initial=
        initial_global_step, desc='Steps', disable=not accelerator.
        is_local_main_process)
    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(t2iadapter):
                if args.pretrained_vae_model_name_or_path is not None:
                    pixel_values = batch['pixel_values'].to(dtype=weight_dtype)
                else:
                    pixel_values = batch['pixel_values']
                latents = []
                for i in range(0, pixel_values.shape[0], 8):
                    latents.append(vae.encode(pixel_values[i:i + 8]).
                        latent_dist.sample())
                latents = torch.cat(latents, dim=0)
                latents = latents * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    latents = latents.to(weight_dtype)
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.rand((bsz,), device=latents.device)
                timesteps = (1 - timesteps ** 3
                    ) * noise_scheduler.config.num_train_timesteps
                timesteps = timesteps.long().to(noise_scheduler.timesteps.dtype
                    )
                timesteps = timesteps.clamp(0, noise_scheduler.config.
                    num_train_timesteps - 1)
                noisy_latents = noise_scheduler.add_noise(latents, noise,
                    timesteps)
                sigmas = get_sigmas(timesteps, len(noisy_latents.shape),
                    noisy_latents.dtype)
                inp_noisy_latents = noisy_latents / (sigmas ** 2 + 1) ** 0.5
                t2iadapter_image = batch['conditioning_pixel_values'].to(dtype
                    =weight_dtype)
                down_block_additional_residuals = t2iadapter(t2iadapter_image)
                down_block_additional_residuals = [sample.to(dtype=
                    weight_dtype) for sample in down_block_additional_residuals
                    ]
                model_pred = unet(inp_noisy_latents, timesteps,
                    encoder_hidden_states=batch['prompt_ids'],
                    added_cond_kwargs=batch['unet_added_conditions'],
                    down_block_additional_residuals=
                    down_block_additional_residuals, return_dict=False)[0]
                denoised_latents = model_pred * -sigmas + noisy_latents
                weighing = sigmas ** -2.0
                if noise_scheduler.config.prediction_type == 'epsilon':
                    target = latents
                elif noise_scheduler.config.prediction_type == 'v_prediction':
                    target = noise_scheduler.get_velocity(latents, noise,
                        timesteps)
                else:
                    raise ValueError(
                        f'Unknown prediction type {noise_scheduler.config.prediction_type}'
                        )
                loss = torch.mean((weighing.float() * (denoised_latents.
                    float() - target.float()) ** 2).reshape(target.shape[0],
                    -1), dim=1)
                loss = loss.mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = t2iadapter.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.
                        max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.
                                startswith('checkpoint')]
                            checkpoints = sorted(checkpoints, key=lambda x:
                                int(x.split('-')[1]))
                            if len(checkpoints
                                ) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints
                                    ) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:
                                    num_to_remove]
                                logger.info(
                                    f'{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints'
                                    )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                    )
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args
                                        .output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        save_path = os.path.join(args.output_dir,
                            f'checkpoint-{global_step}')
                        accelerator.save_state(save_path)
                        logger.info(f'Saved state to {save_path}')
                    if (args.validation_prompt is not None and global_step %
                        args.validation_steps == 0):
                        image_logs = log_validation(vae, unet, t2iadapter,
                            args, accelerator, weight_dtype, global_step)
            logs = {'loss': loss.detach().item(), 'lr': lr_scheduler.
                get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        t2iadapter = unwrap_model(t2iadapter)
        t2iadapter.save_pretrained(args.output_dir)
        if args.push_to_hub:
            save_model_card(repo_id, image_logs=image_logs, base_model=args
                .pretrained_model_name_or_path, repo_folder=args.output_dir)
            upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message='End of training', ignore_patterns=['step_*',
                'epoch_*'])
    accelerator.end_training()
