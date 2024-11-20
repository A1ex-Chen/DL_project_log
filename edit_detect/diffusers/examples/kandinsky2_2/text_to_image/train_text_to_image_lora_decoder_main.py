def main():
    args = parse_args()
    if args.report_to == 'wandb' and args.hub_token is not None:
        raise ValueError(
            'You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token. Please use `huggingface-cli login` to authenticate with the Hub.'
            )
    logging_dir = Path(args.output_dir, args.logging_dir)
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
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
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
    noise_scheduler = DDPMScheduler.from_pretrained(args.
        pretrained_decoder_model_name_or_path, subfolder='scheduler')
    image_processor = CLIPImageProcessor.from_pretrained(args.
        pretrained_prior_model_name_or_path, subfolder='image_processor')
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.
        pretrained_prior_model_name_or_path, subfolder='image_encoder')
    vae = VQModel.from_pretrained(args.
        pretrained_decoder_model_name_or_path, subfolder='movq')
    unet = UNet2DConditionModel.from_pretrained(args.
        pretrained_decoder_model_name_or_path, subfolder='unet')
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith('attn1.processor'
            ) else unet.config.cross_attention_dim
        if name.startswith('mid_block'):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith('up_blocks'):
            block_id = int(name[len('up_blocks.')])
            hidden_size = list(reversed(unet.config.block_out_channels))[
                block_id]
        elif name.startswith('down_blocks'):
            block_id = int(name[len('down_blocks.')])
            hidden_size = unet.config.block_out_channels[block_id]
        lora_attn_procs[name] = LoRAAttnAddedKVProcessor(hidden_size=
            hidden_size, cross_attention_dim=cross_attention_dim, rank=args
            .rank)
    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                'Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`'
                )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(lora_layers.parameters(), lr=args.
        learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name, args.dataset_config_name,
            cache_dir=args.cache_dir)
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files['train'] = os.path.join(args.train_data_dir, '**')
        dataset = load_dataset('imagefolder', data_files=data_files,
            cache_dir=args.cache_dir)
    column_names = dataset['train'].column_names
    image_column = args.image_column
    if image_column not in column_names:
        raise ValueError(
            f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )

    def center_crop(image):
        width, height = image.size
        new_size = min(width, height)
        left = (width - new_size) / 2
        top = (height - new_size) / 2
        right = (width + new_size) / 2
        bottom = (height + new_size) / 2
        return image.crop((left, top, right, bottom))

    def train_transforms(img):
        img = center_crop(img)
        img = img.resize((args.resolution, args.resolution), resample=Image
            .BICUBIC, reducing_gap=1)
        img = np.array(img).astype(np.float32) / 127.5 - 1
        img = torch.from_numpy(np.transpose(img, [2, 0, 1]))
        return img

    def preprocess_train(examples):
        images = [image.convert('RGB') for image in examples[image_column]]
        examples['pixel_values'] = [train_transforms(image) for image in images
            ]
        examples['clip_pixel_values'] = image_processor(images,
            return_tensors='pt').pixel_values
        return examples
    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset['train'] = dataset['train'].shuffle(seed=args.seed).select(
                range(args.max_train_samples))
        train_dataset = dataset['train'].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example['pixel_values'] for example in
            examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format
            ).float()
        clip_pixel_values = torch.stack([example['clip_pixel_values'] for
            example in examples])
        clip_pixel_values = clip_pixel_values.to(memory_format=torch.
            contiguous_format).float()
        return {'pixel_values': pixel_values, 'clip_pixel_values':
            clip_pixel_values}
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
        num_warmup_steps=args.lr_warmup_steps * args.
        gradient_accumulation_steps, num_training_steps=args.
        max_train_steps * args.gradient_accumulation_steps)
    lora_layers, optimizer, train_dataloader, lr_scheduler = (accelerator.
        prepare(lora_layers, optimizer, train_dataloader, lr_scheduler))
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
    args.num_train_epochs = math.ceil(args.max_train_steps /
        num_update_steps_per_epoch)
    if accelerator.is_main_process:
        accelerator.init_trackers('text2image-fine-tune', config=vars(args))
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
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                images = batch['pixel_values'].to(weight_dtype)
                clip_images = batch['clip_pixel_values'].to(weight_dtype)
                latents = vae.encode(images).latents
                image_embeds = image_encoder(clip_images).image_embeds
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.
                    num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise,
                    timesteps)
                target = noise
                added_cond_kwargs = {'image_embeds': image_embeds}
                model_pred = unet(noisy_latents, timesteps, None,
                    added_cond_kwargs=added_cond_kwargs).sample[:, :4]
                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(),
                        reduction='mean')
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma *
                        torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == 'epsilon':
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == 'v_prediction':
                        mse_loss_weights = mse_loss_weights / (snr + 1)
                    loss = F.mse_loss(model_pred.float(), target.float(),
                        reduction='none')
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))
                        ) * mse_loss_weights
                    loss = loss.mean()
                avg_loss = accelerator.gather(loss.repeat(args.
                    train_batch_size)).mean()
                train_loss += avg_loss.item(
                    ) / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.
                        max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({'train_loss': train_loss}, step=global_step)
                train_loss = 0.0
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
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
            logs = {'step_loss': loss.detach().item(), 'lr': lr_scheduler.
                get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if global_step >= args.max_train_steps:
                break
        if accelerator.is_main_process:
            if (args.validation_prompt is not None and epoch % args.
                validation_epochs == 0):
                logger.info(
                    f"""Running validation... 
 Generating {args.num_validation_images} images with prompt: {args.validation_prompt}."""
                    )
                pipeline = AutoPipelineForText2Image.from_pretrained(args.
                    pretrained_decoder_model_name_or_path, unet=accelerator
                    .unwrap_model(unet), torch_dtype=weight_dtype)
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)
                generator = torch.Generator(device=accelerator.device)
                if args.seed is not None:
                    generator = generator.manual_seed(args.seed)
                images = []
                for _ in range(args.num_validation_images):
                    images.append(pipeline(args.validation_prompt,
                        num_inference_steps=30, generator=generator).images[0])
                for tracker in accelerator.trackers:
                    if tracker.name == 'tensorboard':
                        np_images = np.stack([np.asarray(img) for img in
                            images])
                        tracker.writer.add_images('validation', np_images,
                            epoch, dataformats='NHWC')
                    if tracker.name == 'wandb':
                        tracker.log({'validation': [wandb.Image(image,
                            caption=f'{i}: {args.validation_prompt}') for i,
                            image in enumerate(images)]})
                del pipeline
                torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unet.save_attn_procs(args.output_dir)
        if args.push_to_hub:
            save_model_card(repo_id, images=images, base_model=args.
                pretrained_decoder_model_name_or_path, dataset_name=args.
                dataset_name, repo_folder=args.output_dir)
            upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message='End of training', ignore_patterns=['step_*',
                'epoch_*'])
    pipeline = AutoPipelineForText2Image.from_pretrained(args.
        pretrained_decoder_model_name_or_path, torch_dtype=weight_dtype)
    pipeline = pipeline.to(accelerator.device)
    pipeline.unet.load_attn_procs(args.output_dir)
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    images = []
    for _ in range(args.num_validation_images):
        images.append(pipeline(args.validation_prompt, num_inference_steps=
            30, generator=generator).images[0])
    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if len(images) != 0:
                if tracker.name == 'tensorboard':
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images('test', np_images, epoch,
                        dataformats='NHWC')
                if tracker.name == 'wandb':
                    tracker.log({'test': [wandb.Image(image, caption=
                        f'{i}: {args.validation_prompt}') for i, image in
                        enumerate(images)]})
    accelerator.end_training()
