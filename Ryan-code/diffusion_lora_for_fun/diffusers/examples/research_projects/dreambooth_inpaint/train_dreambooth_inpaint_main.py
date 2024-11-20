def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)
    project_config = ProjectConfiguration(total_limit=args.
        checkpoints_total_limit, project_dir=args.output_dir, logging_dir=
        logging_dir)
    accelerator = Accelerator(gradient_accumulation_steps=args.
        gradient_accumulation_steps, mixed_precision=args.mixed_precision,
        log_with='tensorboard', project_config=project_config)
    if (args.train_text_encoder and args.gradient_accumulation_steps > 1 and
        accelerator.num_processes > 1):
        raise ValueError(
            'Gradient accumulation is not supported when training the text encoder in distributed training. Please set gradient_accumulation_steps to 1. This feature will be supported in the future.'
            )
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
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(args.
                pretrained_model_name_or_path, torch_dtype=torch_dtype,
                safety_checker=None)
            pipeline.set_progress_bar_config(disable=True)
            num_new_images = args.num_class_images - cur_class_images
            logger.info(f'Number of class images to sample: {num_new_images}.')
            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset,
                batch_size=args.sample_batch_size, num_workers=1)
            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)
            transform_to_pil = transforms.ToPILImage()
            for example in tqdm(sample_dataloader, desc=
                'Generating class images', disable=not accelerator.
                is_local_main_process):
                bsz = len(example['prompt'])
                fake_images = torch.rand((3, args.resolution, args.resolution))
                transform_to_pil = transforms.ToPILImage()
                fake_pil_images = transform_to_pil(fake_images)
                fake_mask = random_mask((args.resolution, args.resolution),
                    ratio=1, mask_full_image=True)
                images = pipeline(prompt=example['prompt'], mask_image=
                    fake_mask, image=fake_pil_images).images
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
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.
            pretrained_model_name_or_path, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='text_encoder')
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
        subfolder='vae')
    unet = UNet2DConditionModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='unet')
    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()
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
    noise_scheduler = DDPMScheduler.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='scheduler')
    train_dataset = DreamBoothDataset(instance_data_root=args.
        instance_data_dir, instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation
         else None, class_prompt=args.class_prompt, tokenizer=tokenizer,
        size=args.resolution, center_crop=args.center_crop)

    def collate_fn(examples):
        input_ids = [example['instance_prompt_ids'] for example in examples]
        pixel_values = [example['instance_images'] for example in examples]
        if args.with_prior_preservation:
            input_ids += [example['class_prompt_ids'] for example in examples]
            pixel_values += [example['class_images'] for example in examples]
            pior_pil = [example['class_PIL_images'] for example in examples]
        masks = []
        masked_images = []
        for example in examples:
            pil_image = example['PIL_images']
            mask = random_mask(pil_image.size, 1, False)
            mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)
            masks.append(mask)
            masked_images.append(masked_image)
        if args.with_prior_preservation:
            for pil_image in pior_pil:
                mask = random_mask(pil_image.size, 1, False)
                mask, masked_image = prepare_mask_and_masked_image(pil_image,
                    mask)
                masks.append(mask)
                masked_images.append(masked_image)
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format
            ).float()
        input_ids = tokenizer.pad({'input_ids': input_ids}, padding=True,
            return_tensors='pt').input_ids
        masks = torch.stack(masks)
        masked_images = torch.stack(masked_images)
        batch = {'input_ids': input_ids, 'pixel_values': pixel_values,
            'masks': masks, 'masked_images': masked_images}
        return batch
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
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
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = (
            accelerator.prepare(unet, text_encoder, optimizer,
            train_dataloader, lr_scheduler))
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler)
    accelerator.register_for_checkpointing(lr_scheduler)
    weight_dtype = torch.float32
    if args.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif args.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
    args.num_train_epochs = math.ceil(args.max_train_steps /
        num_update_steps_per_epoch)
    if accelerator.is_main_process:
        accelerator.init_trackers('dreambooth', config=vars(args))
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
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            if (args.resume_from_checkpoint and epoch == first_epoch and 
                step < resume_step):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(unet):
                latents = vae.encode(batch['pixel_values'].to(dtype=
                    weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                masked_latents = vae.encode(batch['masked_images'].reshape(
                    batch['pixel_values'].shape).to(dtype=weight_dtype)
                    ).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor
                masks = batch['masks']
                mask = torch.stack([torch.nn.functional.interpolate(mask,
                    size=(args.resolution // 8, args.resolution // 8)) for
                    mask in masks])
                mask = mask.reshape(-1, 1, args.resolution // 8, args.
                    resolution // 8)
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.
                    num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise,
                    timesteps)
                latent_model_input = torch.cat([noisy_latents, mask,
                    masked_latents], dim=1)
                encoder_hidden_states = text_encoder(batch['input_ids'])[0]
                noise_pred = unet(latent_model_input, timesteps,
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
                if args.with_prior_preservation:
                    noise_pred, noise_pred_prior = torch.chunk(noise_pred, 
                        2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    loss = F.mse_loss(noise_pred.float(), target.float(),
                        reduction='none').mean([1, 2, 3]).mean()
                    prior_loss = F.mse_loss(noise_pred_prior.float(),
                        target_prior.float(), reduction='mean')
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(noise_pred.float(), target.float(),
                        reduction='mean')
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(unet.parameters(),
                        text_encoder.parameters()
                        ) if args.train_text_encoder else unet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.
                        max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
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
        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(args.
            pretrained_model_name_or_path, unet=accelerator.unwrap_model(
            unet), text_encoder=accelerator.unwrap_model(text_encoder))
        pipeline.save_pretrained(args.output_dir)
        if args.push_to_hub:
            upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message='End of training', ignore_patterns=['step_*',
                'epoch_*'])
    accelerator.end_training()
