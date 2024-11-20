def main():
    args = parse_args()
    if args.report_to == 'wandb' and args.hub_token is not None:
        raise ValueError(
            'You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token. Please use `huggingface-cli login` to authenticate with the Hub.'
            )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(total_limit=args.
        checkpoints_total_limit, project_dir=args.output_dir, logging_dir=
        logging_dir)
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
        import wandb
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
        tokenizer = MultiTokenCLIPTokenizer.from_pretrained(args.tokenizer_name
            )
    elif args.pretrained_model_name_or_path:
        tokenizer = MultiTokenCLIPTokenizer.from_pretrained(args.
            pretrained_model_name_or_path, subfolder='tokenizer')
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
    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                f'Could not enable memory efficient attention. Make sure xformers is installed correctly and a GPU is available: {e}'
                )
    add_tokens(tokenizer, text_encoder, args.placeholder_token, args.
        num_vec_per_token, args.initializer_token)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    if args.gradient_checkpointing:
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()
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
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.
            gradient_accumulation_steps * args.train_batch_size *
            accelerator.num_processes)
    optimizer = torch.optim.AdamW(text_encoder.get_input_embeddings().
        parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.
        adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon
        )
    train_dataset = TextualInversionDataset(data_root=args.train_data_dir,
        tokenizer=tokenizer, size=args.resolution, placeholder_token=args.
        placeholder_token, repeats=args.repeats, learnable_property=args.
        learnable_property, center_crop=args.center_crop, set='train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.train_batch_size, shuffle=True, num_workers=args.
        dataloader_num_workers)
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes)
    text_encoder, optimizer, train_dataloader, lr_scheduler = (accelerator.
        prepare(text_encoder, optimizer, train_dataloader, lr_scheduler))
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
    args.num_train_epochs = math.ceil(args.max_train_steps /
        num_update_steps_per_epoch)
    if accelerator.is_main_process:
        accelerator.init_trackers('textual_inversion', config=vars(args))
    total_batch_size = (args.train_batch_size * accelerator.num_processes *
        args.gradient_accumulation_steps)
    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
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
        else:
            accelerator.print(f'Resuming from checkpoint {path}')
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split('-')[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch *
                args.gradient_accumulation_steps)
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=
        not accelerator.is_local_main_process)
    progress_bar.set_description('Steps')
    orig_embeds_params = accelerator.unwrap_model(text_encoder
        ).get_input_embeddings().weight.data.clone()
    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            if (args.resume_from_checkpoint and epoch == first_epoch and 
                step < resume_step):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            if args.progressive_tokens:
                train_dataset.prop_tokens_to_load = float(global_step
                    ) / args.progressive_tokens_max_steps
            with accelerator.accumulate(text_encoder):
                latents = vae.encode(batch['pixel_values'].to(dtype=
                    weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.
                    num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise,
                    timesteps)
                encoder_hidden_states = text_encoder(batch['input_ids'])[0].to(
                    dtype=weight_dtype)
                model_pred = unet(noisy_latents, timesteps,
                    encoder_hidden_states).sample
                if noise_scheduler.config.prediction_type == 'epsilon':
                    target = noise
                elif noise_scheduler.config.prediction_type == 'v_prediction':
                    target = noise_scheduler.get_velocity(latents, noise,
                        timesteps)
                else:
                    raise ValueError(
                        f'Unknown prediction type {noise_scheduler.config.prediction_type}'
                        )
                loss = F.mse_loss(model_pred.float(), target.float(),
                    reduction='mean')
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                index_no_updates = get_mask(tokenizer, accelerator)
                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder
                        ).get_input_embeddings().weight[index_no_updates
                        ] = orig_embeds_params[index_no_updates]
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir,
                        f'learned_embeds-steps-{global_step}.bin')
                    save_progress(tokenizer, text_encoder, accelerator,
                        save_path)
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir,
                            f'checkpoint-{global_step}')
                        accelerator.save_state(save_path)
                        logger.info(f'Saved state to {save_path}')
            logs = {'loss': loss.detach().item(), 'lr': lr_scheduler.
                get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
        if (accelerator.is_main_process and args.validation_prompt is not
            None and epoch % args.validation_epochs == 0):
            logger.info(
                f"""Running validation... 
 Generating {args.num_validation_images} images with prompt: {args.validation_prompt}."""
                )
            pipeline = DiffusionPipeline.from_pretrained(args.
                pretrained_model_name_or_path, text_encoder=accelerator.
                unwrap_model(text_encoder), tokenizer=tokenizer, unet=unet,
                vae=vae, revision=args.revision, torch_dtype=weight_dtype)
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                pipeline.scheduler.config)
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)
            generator = None if args.seed is None else torch.Generator(device
                =accelerator.device).manual_seed(args.seed)
            images = []
            for _ in range(args.num_validation_images):
                with torch.autocast('cuda'):
                    image = pipeline(args.validation_prompt,
                        num_inference_steps=25, generator=generator).images[0]
                images.append(image)
            for tracker in accelerator.trackers:
                if tracker.name == 'tensorboard':
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images('validation', np_images,
                        epoch, dataformats='NHWC')
                if tracker.name == 'wandb':
                    tracker.log({'validation': [wandb.Image(image, caption=
                        f'{i}: {args.validation_prompt}') for i, image in
                        enumerate(images)]})
            del pipeline
            torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.push_to_hub and args.only_save_embeds:
            logger.warning(
                'Enabling full model saving because --push_to_hub=True was specified.'
                )
            save_full_model = True
        else:
            save_full_model = not args.only_save_embeds
        if save_full_model:
            pipeline = StableDiffusionPipeline.from_pretrained(args.
                pretrained_model_name_or_path, text_encoder=accelerator.
                unwrap_model(text_encoder), vae=vae, unet=unet, tokenizer=
                tokenizer)
            pipeline.save_pretrained(args.output_dir)
        save_path = os.path.join(args.output_dir, 'learned_embeds.bin')
        save_progress(tokenizer, text_encoder, accelerator, save_path)
        if args.push_to_hub:
            upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message='End of training', ignore_patterns=['step_*',
                'epoch_*'])
    accelerator.end_training()
