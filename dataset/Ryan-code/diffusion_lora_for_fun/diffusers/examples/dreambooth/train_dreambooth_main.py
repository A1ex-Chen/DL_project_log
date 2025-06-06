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
    if args.report_to == 'wandb':
        if not is_wandb_available():
            raise ImportError(
                'Make sure to install wandb if you want to use it for logging during training.'
                )
    if (args.train_text_encoder and args.gradient_accumulation_steps > 1 and
        accelerator.num_processes > 1):
        raise ValueError(
            'Gradient accumulation is not supported when training the text encoder in distributed training. Please set gradient_accumulation_steps to 1. This feature will be supported in the future.'
            )
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
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))
        if cur_class_images < args.num_class_images:
            torch_dtype = (torch.float16 if accelerator.device.type ==
                'cuda' else torch.float32)
            if args.prior_generation_precision == 'fp32':
                torch_dtype = torch.float32
            elif args.prior_generation_precision == 'fp16':
                torch_dtype = torch.float16
            elif args.prior_generation_precision == 'bf16':
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(args.
                pretrained_model_name_or_path, torch_dtype=torch_dtype,
                safety_checker=None, revision=args.revision, variant=args.
                variant)
            pipeline.set_progress_bar_config(disable=True)
            num_new_images = args.num_class_images - cur_class_images
            logger.info(f'Number of class images to sample: {num_new_images}.')
            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset,
                batch_size=args.sample_batch_size)
            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)
            for example in tqdm(sample_dataloader, desc=
                'Generating class images', disable=not accelerator.
                is_local_main_process):
                images = pipeline(example['prompt']).images
                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()
                        ).hexdigest()
                    image_filename = (class_images_dir /
                        f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        )
                    image.save(image_filename)
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.
                output_dir).name, exist_ok=True, token=args.hub_token).repo_id
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name,
            revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.
            pretrained_model_name_or_path, subfolder='tokenizer', revision=
            args.revision, use_fast=False)
    text_encoder_cls = import_model_class_from_model_name_or_path(args.
        pretrained_model_name_or_path, args.revision)
    noise_scheduler = DDPMScheduler.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='scheduler')
    text_encoder = text_encoder_cls.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='text_encoder', revision=
        args.revision, variant=args.variant)
    if model_has_vae(args):
        vae = AutoencoderKL.from_pretrained(args.
            pretrained_model_name_or_path, subfolder='vae', revision=args.
            revision, variant=args.variant)
    else:
        vae = None
    unet = UNet2DConditionModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='unet', revision=args.
        revision, variant=args.variant)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                sub_dir = 'unet' if isinstance(model, type(unwrap_model(unet))
                    ) else 'text_encoder'
                model.save_pretrained(os.path.join(output_dir, sub_dir))
                weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(text_encoder))):
                load_model = text_encoder_cls.from_pretrained(input_dir,
                    subfolder='text_encoder')
                model.config = load_model.config
            else:
                load_model = UNet2DConditionModel.from_pretrained(input_dir,
                    subfolder='unet')
                model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict())
            del load_model
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    if vae is not None:
        vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)
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
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()
    low_precision_error_string = (
        'Please make sure to always have all model weights in full float32 precision when starting training - even if doing mixed precision training. copy of the weights should still be float32.'
        )
    if unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f'Unet loaded as datatype {unwrap_model(unet).dtype}. {low_precision_error_string}'
            )
    if args.train_text_encoder and unwrap_model(text_encoder
        ).dtype != torch.float32:
        raise ValueError(
            f'Text encoder loaded as datatype {unwrap_model(text_encoder).dtype}. {low_precision_error_string}'
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
    params_to_optimize = itertools.chain(unet.parameters(), text_encoder.
        parameters()) if args.train_text_encoder else unet.parameters()
    optimizer = optimizer_class(params_to_optimize, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.
        adam_weight_decay, eps=args.adam_epsilon)
    if args.pre_compute_text_embeddings:

        def compute_text_embeddings(prompt):
            with torch.no_grad():
                text_inputs = tokenize_prompt(tokenizer, prompt,
                    tokenizer_max_length=args.tokenizer_max_length)
                prompt_embeds = encode_prompt(text_encoder, text_inputs.
                    input_ids, text_inputs.attention_mask,
                    text_encoder_use_attention_mask=args.
                    text_encoder_use_attention_mask)
            return prompt_embeds
        pre_computed_encoder_hidden_states = compute_text_embeddings(args.
            instance_prompt)
        validation_prompt_negative_prompt_embeds = compute_text_embeddings('')
        if args.validation_prompt is not None:
            validation_prompt_encoder_hidden_states = compute_text_embeddings(
                args.validation_prompt)
        else:
            validation_prompt_encoder_hidden_states = None
        if args.class_prompt is not None:
            pre_computed_class_prompt_encoder_hidden_states = (
                compute_text_embeddings(args.class_prompt))
        else:
            pre_computed_class_prompt_encoder_hidden_states = None
        text_encoder = None
        tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
    else:
        pre_computed_encoder_hidden_states = None
        validation_prompt_encoder_hidden_states = None
        validation_prompt_negative_prompt_embeds = None
        pre_computed_class_prompt_encoder_hidden_states = None
    train_dataset = DreamBoothDataset(instance_data_root=args.
        instance_data_dir, instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation
         else None, class_prompt=args.class_prompt, class_num=args.
        num_class_images, tokenizer=tokenizer, size=args.resolution,
        center_crop=args.center_crop, encoder_hidden_states=
        pre_computed_encoder_hidden_states,
        class_prompt_encoder_hidden_states=
        pre_computed_class_prompt_encoder_hidden_states,
        tokenizer_max_length=args.tokenizer_max_length)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.train_batch_size, shuffle=True, collate_fn=lambda
        examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers)
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = (
            accelerator.prepare(unet, text_encoder, optimizer,
            train_dataloader, lr_scheduler))
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    if vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder and text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
    args.num_train_epochs = math.ceil(args.max_train_steps /
        num_update_steps_per_epoch)
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        tracker_config.pop('validation_images')
        accelerator.init_trackers('dreambooth', config=tracker_config)
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
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch['pixel_values'].to(dtype=weight_dtype)
                if vae is not None:
                    model_input = vae.encode(batch['pixel_values'].to(dtype
                        =weight_dtype)).latent_dist.sample()
                    model_input = model_input * vae.config.scaling_factor
                else:
                    model_input = pixel_values
                if args.offset_noise:
                    noise = torch.randn_like(model_input) + 0.1 * torch.randn(
                        model_input.shape[0], model_input.shape[1], 1, 1,
                        device=model_input.device)
                else:
                    noise = torch.randn_like(model_input)
                bsz, channels, height, width = model_input.shape
                timesteps = torch.randint(0, noise_scheduler.config.
                    num_train_timesteps, (bsz,), device=model_input.device)
                timesteps = timesteps.long()
                noisy_model_input = noise_scheduler.add_noise(model_input,
                    noise, timesteps)
                if args.pre_compute_text_embeddings:
                    encoder_hidden_states = batch['input_ids']
                else:
                    encoder_hidden_states = encode_prompt(text_encoder,
                        batch['input_ids'], batch['attention_mask'],
                        text_encoder_use_attention_mask=args.
                        text_encoder_use_attention_mask)
                if unwrap_model(unet).config.in_channels == channels * 2:
                    noisy_model_input = torch.cat([noisy_model_input,
                        noisy_model_input], dim=1)
                if args.class_labels_conditioning == 'timesteps':
                    class_labels = timesteps
                else:
                    class_labels = None
                model_pred = unet(noisy_model_input, timesteps,
                    encoder_hidden_states, class_labels=class_labels,
                    return_dict=False)[0]
                if model_pred.shape[1] == 6:
                    model_pred, _ = torch.chunk(model_pred, 2, dim=1)
                if noise_scheduler.config.prediction_type == 'epsilon':
                    target = noise
                elif noise_scheduler.config.prediction_type == 'v_prediction':
                    target = noise_scheduler.get_velocity(model_input,
                        noise, timesteps)
                else:
                    raise ValueError(
                        f'Unknown prediction type {noise_scheduler.config.prediction_type}'
                        )
                if args.with_prior_preservation:
                    model_pred, model_pred_prior = torch.chunk(model_pred, 
                        2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    prior_loss = F.mse_loss(model_pred_prior.float(),
                        target_prior.float(), reduction='mean')
                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(),
                        reduction='mean')
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    base_weight = torch.stack([snr, args.snr_gamma * torch.
                        ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    if (noise_scheduler.config.prediction_type ==
                        'v_prediction'):
                        mse_loss_weights = base_weight + 1
                    else:
                        mse_loss_weights = base_weight
                    loss = F.mse_loss(model_pred.float(), target.float(),
                        reduction='none')
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))
                        ) * mse_loss_weights
                    loss = loss.mean()
                if args.with_prior_preservation:
                    loss = loss + args.prior_loss_weight * prior_loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(unet.parameters(),
                        text_encoder.parameters()
                        ) if args.train_text_encoder else unet.parameters()
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
                    images = []
                    if (args.validation_prompt is not None and global_step %
                        args.validation_steps == 0):
                        images = log_validation(unwrap_model(text_encoder) if
                            text_encoder is not None else text_encoder,
                            tokenizer, unwrap_model(unet), vae, args,
                            accelerator, weight_dtype, global_step,
                            validation_prompt_encoder_hidden_states,
                            validation_prompt_negative_prompt_embeds)
            logs = {'loss': loss.detach().item(), 'lr': lr_scheduler.
                get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline_args = {}
        if text_encoder is not None:
            pipeline_args['text_encoder'] = unwrap_model(text_encoder)
        if args.skip_save_text_encoder:
            pipeline_args['text_encoder'] = None
        pipeline = DiffusionPipeline.from_pretrained(args.
            pretrained_model_name_or_path, unet=unwrap_model(unet),
            revision=args.revision, variant=args.variant, **pipeline_args)
        scheduler_args = {}
        if 'variance_type' in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type
            if variance_type in ['learned', 'learned_range']:
                variance_type = 'fixed_small'
            scheduler_args['variance_type'] = variance_type
        pipeline.scheduler = pipeline.scheduler.from_config(pipeline.
            scheduler.config, **scheduler_args)
        pipeline.save_pretrained(args.output_dir)
        if args.push_to_hub:
            save_model_card(repo_id, images=images, base_model=args.
                pretrained_model_name_or_path, train_text_encoder=args.
                train_text_encoder, prompt=args.instance_prompt,
                repo_folder=args.output_dir, pipeline=pipeline)
            upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message='End of training', ignore_patterns=['step_*',
                'epoch_*'])
    accelerator.end_training()
