def main():
    args = parse_args()
    project_config = ProjectConfiguration(total_limit=args.
        checkpoints_total_limit, project_dir=args.output_dir, logging_dir=
        Path(args.output_dir, args.logging_dir))
    accelerator = Accelerator(gradient_accumulation_steps=args.
        gradient_accumulation_steps, mixed_precision=args.mixed_precision,
        project_config=project_config, log_with='wandb' if args.
        report_to_wandb else None)
    if args.report_to_wandb and not is_wandb_available():
        raise ImportError(
            'Make sure to install wandb if you want to use it for logging during training.'
            )
    if args.seed is not None:
        set_seed(args.seed)
    logging.basicConfig(format=
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt=
        '%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    tokenizer = CLIPTokenizer.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='text_encoder'
        ).requires_grad_(args.train_text_encoder)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
        subfolder='vae').requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='unet')
    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.
            gradient_accumulation_steps * args.train_batch_size *
            accelerator.num_processes)
    optimizer = torch.optim.AdamW(params=itertools.chain(unet.parameters(),
        text_encoder.parameters()) if args.train_text_encoder else unet.
        parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.
        adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon
        )
    noise_scheduler = DDPMScheduler.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='scheduler')
    train_dataset = DreamBoothDataset(tokenizer=tokenizer, datasets_paths=
        args.instance_data_dir)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.train_batch_size, shuffle=True, collate_fn=lambda
        examples: collate_fn(examples, tokenizer))
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
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
    if args.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif args.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps /
        num_update_steps_per_epoch)
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(args.validation_project_name, config=
            tracker_config)
    val_pipeline = StableDiffusionInpaintPipeline.from_pretrained(args.
        pretrained_model_name_or_path, tokenizer=tokenizer, text_encoder=
        text_encoder, unet=unet, vae=vae, torch_dtype=weight_dtype,
        safety_checker=None)
    val_pipeline.set_progress_bar_config(disable=True)
    val_pairs = [{'image': example['image'], 'mask_image': mask, 'prompt':
        example['prompt']} for example in train_dataset.test_data for mask in
        [example[key] for key in example if 'mask' in key]]

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                sub_dir = 'unet' if isinstance(model, type(accelerator.
                    unwrap_model(unet))) else 'text_encoder'
                model.save_pretrained(os.path.join(output_dir, sub_dir))
                weights.pop()
    accelerator.register_save_state_pre_hook(save_model_hook)
    print()
    total_batch_size = (args.train_batch_size * accelerator.num_processes *
        args.gradient_accumulation_steps)
    logger.info('***** Running training *****')
    logger.info(f'  Num batches each epoch = {len(train_dataloader)}')
    logger.info(f'  Num Epochs = {num_train_epochs}')
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
    for epoch in range(first_epoch, num_train_epochs):
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
                if accelerator.is_main_process:
                    if (global_step % args.validation_steps == 0 and 
                        global_step >= args.validation_from and args.
                        report_to_wandb):
                        log_validation(val_pipeline, text_encoder, unet,
                            val_pairs, accelerator)
                    if (global_step % args.checkpointing_steps == 0 and 
                        global_step >= args.checkpointing_from):
                        checkpoint(args, global_step, accelerator)
            logs = {'loss': loss.detach().item(), 'lr': lr_scheduler.
                get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
        accelerator.wait_for_everyone()
    accelerator.end_training()
