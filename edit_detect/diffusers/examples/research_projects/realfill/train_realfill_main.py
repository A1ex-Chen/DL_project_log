def main(args):
    if args.report_to == 'wandb' and args.hub_token is not None:
        raise ValueError(
            'You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token. Please use `huggingface-cli login` to authenticate with the Hub.'
            )
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(gradient_accumulation_steps=args.
        gradient_accumulation_steps, mixed_precision=args.mixed_precision,
        log_with=args.report_to, project_dir=logging_dir)
    if args.report_to == 'wandb':
        if not is_wandb_available():
            raise ImportError(
                'Make sure to install wandb if you want to use it for logging during training.'
                )
        wandb.login(key=args.wandb_key)
        wandb.init(project=args.wandb_project_name)
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
                output_dir).name, exist_ok=True, token=args.hub_token).repo_id
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name,
            revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.
            pretrained_model_name_or_path, subfolder='tokenizer', revision=
            args.revision, use_fast=False)
    noise_scheduler = DDPMScheduler.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='scheduler')
    text_encoder = CLIPTextModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='text_encoder', revision=
        args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
        subfolder='vae', revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='unet', revision=args.revision
        )
    config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules=['to_k', 'to_q', 'to_v', 'key', 'query', 'value'],
        lora_dropout=args.lora_dropout, bias=args.lora_bias)
    unet = get_peft_model(unet, config)
    config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules=['k_proj', 'q_proj', 'v_proj'], lora_dropout=args.
        lora_dropout, bias=args.lora_bias)
    text_encoder = get_peft_model(text_encoder, config)
    vae.requires_grad_(False)
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
        text_encoder.gradient_checkpointing_enable()

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                sub_dir = 'unet' if isinstance(model.base_model.model, type
                    (accelerator.unwrap_model(unet).base_model.model)
                    ) else 'text_encoder'
                model.save_pretrained(os.path.join(output_dir, sub_dir))
                weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            model = models.pop()
            sub_dir = 'unet' if isinstance(model.base_model.model, type(
                accelerator.unwrap_model(unet).base_model.model)
                ) else 'text_encoder'
            model_cls = UNet2DConditionModel if isinstance(model.base_model
                .model, type(accelerator.unwrap_model(unet).base_model.model)
                ) else CLIPTextModel
            load_model = model_cls.from_pretrained(args.
                pretrained_model_name_or_path, subfolder=sub_dir)
            load_model = PeftModel.from_pretrained(load_model, input_dir,
                subfolder=sub_dir)
            model.load_state_dict(load_model.state_dict())
            del load_model
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.scale_lr:
        args.unet_learning_rate = (args.unet_learning_rate * args.
            gradient_accumulation_steps * args.train_batch_size *
            accelerator.num_processes)
        args.text_encoder_learning_rate = (args.text_encoder_learning_rate *
            args.gradient_accumulation_steps * args.train_batch_size *
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
    optimizer = optimizer_class([{'params': unet.parameters(), 'lr': args.
        unet_learning_rate}, {'params': text_encoder.parameters(), 'lr':
        args.text_encoder_learning_rate}], betas=(args.adam_beta1, args.
        adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon
        )
    train_dataset = RealFillDataset(train_data_root=args.train_data_dir,
        tokenizer=tokenizer, size=args.resolution)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.train_batch_size, shuffle=True, collate_fn=
        collate_fn, num_workers=1)
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.
        gradient_accumulation_steps, num_training_steps=args.
        max_train_steps * args.gradient_accumulation_steps, num_cycles=args
        .lr_num_cycles, power=args.lr_power)
    unet, text_encoder, optimizer, train_dataloader = accelerator.prepare(unet,
        text_encoder, optimizer, train_dataloader)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
    args.num_train_epochs = math.ceil(args.max_train_steps /
        num_update_steps_per_epoch)
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers('realfill', config=tracker_config)
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
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, text_encoder):
                latents = vae.encode(batch['images'].to(dtype=weight_dtype)
                    ).latent_dist.sample()
                latents = latents * 0.18215
                conditionings = vae.encode(batch['conditioning_images'].to(
                    dtype=weight_dtype)).latent_dist.sample()
                conditionings = conditionings * 0.18215
                masks, size = batch['masks'].to(dtype=weight_dtype
                    ), latents.shape[2:]
                masks = F.interpolate(masks, size=size)
                weightings = batch['weightings'].to(dtype=weight_dtype)
                weightings = F.interpolate(weightings, size=size)
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.
                    num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise,
                    timesteps)
                inputs = torch.cat([noisy_latents, masks, conditionings], dim=1
                    )
                encoder_hidden_states = text_encoder(batch['input_ids'])[0]
                model_pred = unet(inputs, timesteps, encoder_hidden_states
                    ).sample
                assert noise_scheduler.config.prediction_type == 'epsilon'
                loss = (weightings * F.mse_loss(model_pred.float(), noise.
                    float(), reduction='none')).mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(unet.parameters(),
                        text_encoder.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.
                        max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                if args.report_to == 'wandb':
                    accelerator.print(progress_bar)
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
                    if global_step % args.validation_steps == 0:
                        log_validation(text_encoder, tokenizer, unet, args,
                            accelerator, weight_dtype, global_step)
            logs = {'loss': loss.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(args.
            pretrained_model_name_or_path, unet=accelerator.unwrap_model(
            unet, keep_fp32_wrapper=True).merge_and_unload(), text_encoder=
            accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True).
            merge_and_unload(), revision=args.revision)
        pipeline.save_pretrained(args.output_dir)
        images = log_validation(text_encoder, tokenizer, unet, args,
            accelerator, weight_dtype, global_step)
        if args.push_to_hub:
            save_model_card(repo_id, images=images, base_model=args.
                pretrained_model_name_or_path, repo_folder=args.output_dir)
            upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message='End of training', ignore_patterns=['step_*',
                'epoch_*'])
    accelerator.end_training()
