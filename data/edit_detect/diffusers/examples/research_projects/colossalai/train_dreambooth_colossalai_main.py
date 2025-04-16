def main(args):
    if args.seed is None:
        colossalai.launch_from_torch(config={})
    else:
        colossalai.launch_from_torch(config={}, seed=args.seed)
    local_rank = gpc.get_local_rank(ParallelMode.DATA)
    world_size = gpc.get_world_size(ParallelMode.DATA)
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))
        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if get_current_device(
                ) == 'cuda' else torch.float32
            pipeline = DiffusionPipeline.from_pretrained(args.
                pretrained_model_name_or_path, torch_dtype=torch_dtype,
                safety_checker=None, revision=args.revision)
            pipeline.set_progress_bar_config(disable=True)
            num_new_images = args.num_class_images - cur_class_images
            logger.info(f'Number of class images to sample: {num_new_images}.')
            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset,
                batch_size=args.sample_batch_size)
            pipeline.to(get_current_device())
            for example in tqdm(sample_dataloader, desc=
                'Generating class images', disable=not local_rank == 0):
                images = pipeline(example['prompt']).images
                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()
                        ).hexdigest()
                    image_filename = (class_images_dir /
                        f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        )
                    image.save(image_filename)
            del pipeline
    if local_rank == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.
                output_dir).name, exist_ok=True, token=args.hub_token).repo_id
    if args.tokenizer_name:
        logger.info(f'Loading tokenizer from {args.tokenizer_name}', ranks=[0])
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name,
            revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        logger.info('Loading tokenizer from pretrained model', ranks=[0])
        tokenizer = AutoTokenizer.from_pretrained(args.
            pretrained_model_name_or_path, subfolder='tokenizer', revision=
            args.revision, use_fast=False)
    text_encoder_cls = import_model_class_from_model_name_or_path(args.
        pretrained_model_name_or_path)
    logger.info(
        f'Loading text_encoder from {args.pretrained_model_name_or_path}',
        ranks=[0])
    text_encoder = text_encoder_cls.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='text_encoder', revision=
        args.revision)
    logger.info(
        f'Loading AutoencoderKL from {args.pretrained_model_name_or_path}',
        ranks=[0])
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
        subfolder='vae', revision=args.revision)
    logger.info(
        f'Loading UNet2DConditionModel from {args.pretrained_model_name_or_path}'
        , ranks=[0])
    with ColoInitContext(device=get_current_device()):
        unet = UNet2DConditionModel.from_pretrained(args.
            pretrained_model_name_or_path, subfolder='unet', revision=args.
            revision, low_cpu_mem_usage=False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.train_batch_size *
            world_size)
    unet = gemini_zero_dpp(unet, args.placement)
    optimizer = GeminiAdamOptimizer(unet, lr=args.learning_rate,
        initial_scale=2 ** 5, clipping_norm=args.max_grad_norm)
    noise_scheduler = DDPMScheduler.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='scheduler')
    logger.info(f'Prepare dataset from {args.instance_data_dir}', ranks=[0])
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
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format
            ).float()
        input_ids = tokenizer.pad({'input_ids': input_ids}, padding=
            'max_length', max_length=tokenizer.model_max_length,
            return_tensors='pt').input_ids
        batch = {'input_ids': input_ids, 'pixel_values': pixel_values}
        return batch
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.train_batch_size, shuffle=True, collate_fn=
        collate_fn, num_workers=1)
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    if args.max_train_steps is None:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps, num_training_steps=args.
        max_train_steps)
    weight_dtype = torch.float32
    if args.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif args.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    vae.to(get_current_device(), dtype=weight_dtype)
    text_encoder.to(get_current_device(), dtype=weight_dtype)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    if overrode_max_train_steps:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
    args.num_train_epochs = math.ceil(args.max_train_steps /
        num_update_steps_per_epoch)
    total_batch_size = args.train_batch_size * world_size
    logger.info('***** Running training *****', ranks=[0])
    logger.info(f'  Num examples = {len(train_dataset)}', ranks=[0])
    logger.info(f'  Num batches each epoch = {len(train_dataloader)}',
        ranks=[0])
    logger.info(f'  Num Epochs = {args.num_train_epochs}', ranks=[0])
    logger.info(
        f'  Instantaneous batch size per device = {args.train_batch_size}',
        ranks=[0])
    logger.info(
        f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}'
        , ranks=[0])
    logger.info(f'  Total optimization steps = {args.max_train_steps}',
        ranks=[0])
    progress_bar = tqdm(range(args.max_train_steps), disable=not local_rank ==
        0)
    progress_bar.set_description('Steps')
    global_step = 0
    torch.cuda.synchronize()
    for epoch in range(args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            torch.cuda.reset_peak_memory_stats()
            for key, value in batch.items():
                batch[key] = value.to(get_current_device(), non_blocking=True)
            optimizer.zero_grad()
            latents = vae.encode(batch['pixel_values'].to(dtype=weight_dtype)
                ).latent_dist.sample()
            latents = latents * 0.18215
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.
                num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps
                )
            encoder_hidden_states = text_encoder(batch['input_ids'])[0]
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states
                ).sample
            if noise_scheduler.config.prediction_type == 'epsilon':
                target = noise
            elif noise_scheduler.config.prediction_type == 'v_prediction':
                target = noise_scheduler.get_velocity(latents, noise, timesteps
                    )
            else:
                raise ValueError(
                    f'Unknown prediction type {noise_scheduler.config.prediction_type}'
                    )
            if args.with_prior_preservation:
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0
                    )
                target, target_prior = torch.chunk(target, 2, dim=0)
                loss = F.mse_loss(model_pred.float(), target.float(),
                    reduction='none').mean([1, 2, 3]).mean()
                prior_loss = F.mse_loss(model_pred_prior.float(),
                    target_prior.float(), reduction='mean')
                loss = loss + args.prior_loss_weight * prior_loss
            else:
                loss = F.mse_loss(model_pred.float(), target.float(),
                    reduction='mean')
            optimizer.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            logger.info(
                f'max GPU_mem cost is {torch.cuda.max_memory_allocated() / 2 ** 20} MB'
                , ranks=[0])
            progress_bar.update(1)
            global_step += 1
            logs = {'loss': loss.detach().item(), 'lr': optimizer.
                param_groups[0]['lr']}
            progress_bar.set_postfix(**logs)
            if global_step % args.save_steps == 0:
                torch.cuda.synchronize()
                torch_unet = get_static_torch_model(unet)
                if local_rank == 0:
                    pipeline = DiffusionPipeline.from_pretrained(args.
                        pretrained_model_name_or_path, unet=torch_unet,
                        revision=args.revision)
                    save_path = os.path.join(args.output_dir,
                        f'checkpoint-{global_step}')
                    pipeline.save_pretrained(save_path)
                    logger.info(f'Saving model checkpoint to {save_path}',
                        ranks=[0])
            if global_step >= args.max_train_steps:
                break
    torch.cuda.synchronize()
    unet = get_static_torch_model(unet)
    if local_rank == 0:
        pipeline = DiffusionPipeline.from_pretrained(args.
            pretrained_model_name_or_path, unet=unet, revision=args.revision)
        pipeline.save_pretrained(args.output_dir)
        logger.info(f'Saving model checkpoint to {args.output_dir}', ranks=[0])
        if args.push_to_hub:
            upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message='End of training', ignore_patterns=['step_*',
                'epoch_*'])
