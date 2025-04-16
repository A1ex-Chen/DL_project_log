def main(args):
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
    if args.logger == 'tensorboard':
        if not is_tensorboard_available():
            raise ImportError(
                'Make sure to install tensorboard if you want to use it for logging during training.'
                )
    elif args.logger == 'wandb':
        if not is_wandb_available():
            raise ImportError(
                'Make sure to install wandb if you want to use it for logging during training.'
                )
        import wandb
    if version.parse(accelerate.__version__) >= version.parse('0.16.0'):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir,
                        'unet_ema'))
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, 'unet'))
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(
                    input_dir, 'unet_ema'), UNet2DModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model
            for i in range(len(models)):
                model = models.pop()
                load_model = UNet2DModel.from_pretrained(input_dir,
                    subfolder='unet')
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    logging.basicConfig(format=
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt=
        '%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.
                output_dir).name, exist_ok=True, token=args.hub_token).repo_id
    if args.model_config_name_or_path is None:
        model = UNet2DModel(sample_size=args.resolution, in_channels=3,
            out_channels=3, layers_per_block=2, block_out_channels=(128, 
            128, 256, 256, 512, 512), down_block_types=('DownBlock2D',
            'DownBlock2D', 'DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D',
            'DownBlock2D'), up_block_types=('UpBlock2D', 'AttnUpBlock2D',
            'UpBlock2D', 'UpBlock2D', 'UpBlock2D', 'UpBlock2D'))
    else:
        config = UNet2DModel.load_config(args.model_config_name_or_path)
        model = UNet2DModel.from_config(config)
    if args.use_ema:
        ema_model = EMAModel(model.parameters(), decay=args.ema_max_decay,
            use_ema_warmup=True, inv_gamma=args.ema_inv_gamma, power=args.
            ema_power, model_cls=UNet2DModel, model_config=model.config)
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse('0.0.16'):
                logger.warning(
                    'xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details.'
                    )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                'xformers is not available. Make sure it is installed correctly'
                )
    accepts_prediction_type = 'prediction_type' in set(inspect.signature(
        DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.
            ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type)
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.
            ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.
        adam_weight_decay, eps=args.adam_epsilon)
    optimizer = ORT_FP16_Optimizer(optimizer)
    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name, args.dataset_config_name,
            cache_dir=args.cache_dir, split='train')
    else:
        dataset = load_dataset('imagefolder', data_dir=args.train_data_dir,
            cache_dir=args.cache_dir, split='train')
    augmentations = transforms.Compose([transforms.Resize(args.resolution,
        interpolation=transforms.InterpolationMode.BILINEAR), transforms.
        CenterCrop(args.resolution) if args.center_crop else transforms.
        RandomCrop(args.resolution), transforms.RandomHorizontalFlip() if
        args.random_flip else transforms.Lambda(lambda x: x), transforms.
        ToTensor(), transforms.Normalize([0.5], [0.5])])

    def transform_images(examples):
        images = [augmentations(image.convert('RGB')) for image in examples
            ['image']]
        return {'input': images}
    logger.info(f'Dataset size: {len(dataset)}')
    dataset.set_transform(transform_images)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args
        .train_batch_size, shuffle=True, num_workers=args.
        dataloader_num_workers)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.
        gradient_accumulation_steps, num_training_steps=len(
        train_dataloader) * args.num_epochs)
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)
    if args.use_ema:
        ema_model.to(accelerator.device)
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split('.')[0]
        accelerator.init_trackers(run)
    model = ORTModule(model)
    total_batch_size = (args.train_batch_size * accelerator.num_processes *
        args.gradient_accumulation_steps)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(dataset)}')
    logger.info(f'  Num Epochs = {args.num_epochs}')
    logger.info(
        f'  Instantaneous batch size per device = {args.train_batch_size}')
    logger.info(
        f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}'
        )
    logger.info(
        f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}')
    logger.info(f'  Total optimization steps = {max_train_steps}')
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
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not
            accelerator.is_local_main_process)
        progress_bar.set_description(f'Epoch {epoch}')
        for step, batch in enumerate(train_dataloader):
            if (args.resume_from_checkpoint and epoch == first_epoch and 
                step < resume_step):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            clean_images = batch['input']
            noise = torch.randn(clean_images.shape, dtype=torch.float32 if 
                args.mixed_precision == 'no' else torch.float16).to(
                clean_images.device)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.
                num_train_timesteps, (bsz,), device=clean_images.device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise,
                timesteps)
            with accelerator.accumulate(model):
                model_output = model(noisy_images, timesteps, return_dict=False
                    )[0]
                if args.prediction_type == 'epsilon':
                    loss = F.mse_loss(model_output, noise)
                elif args.prediction_type == 'sample':
                    alpha_t = _extract_into_tensor(noise_scheduler.
                        alphas_cumprod, timesteps, (clean_images.shape[0], 
                        1, 1, 1))
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = snr_weights * F.mse_loss(model_output,
                        clean_images, reduction='none')
                    loss = loss.mean()
                else:
                    raise ValueError(
                        f'Unsupported prediction type: {args.prediction_type}')
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir,
                            f'checkpoint-{global_step}')
                        accelerator.save_state(save_path)
                        logger.info(f'Saved state to {save_path}')
            logs = {'loss': loss.detach().item(), 'lr': lr_scheduler.
                get_last_lr()[0], 'step': global_step}
            if args.use_ema:
                logs['ema_decay'] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if (epoch % args.save_images_epochs == 0 or epoch == args.
                num_epochs - 1):
                unet = accelerator.unwrap_model(model)
                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())
                pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
                generator = torch.Generator(device=pipeline.device
                    ).manual_seed(0)
                images = pipeline(generator=generator, batch_size=args.
                    eval_batch_size, num_inference_steps=args.
                    ddpm_num_inference_steps, output_type='np').images
                if args.use_ema:
                    ema_model.restore(unet.parameters())
                images_processed = (images * 255).round().astype('uint8')
                if args.logger == 'tensorboard':
                    if is_accelerate_version('>=', '0.17.0.dev0'):
                        tracker = accelerator.get_tracker('tensorboard',
                            unwrap=True)
                    else:
                        tracker = accelerator.get_tracker('tensorboard')
                    tracker.add_images('test_samples', images_processed.
                        transpose(0, 3, 1, 2), epoch)
                elif args.logger == 'wandb':
                    accelerator.get_tracker('wandb').log({'test_samples': [
                        wandb.Image(img) for img in images_processed],
                        'epoch': epoch}, step=global_step)
            if (epoch % args.save_model_epochs == 0 or epoch == args.
                num_epochs - 1):
                unet = accelerator.unwrap_model(model)
                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())
                pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
                pipeline.save_pretrained(args.output_dir)
                if args.use_ema:
                    ema_model.restore(unet.parameters())
                if args.push_to_hub:
                    upload_folder(repo_id=repo_id, folder_path=args.
                        output_dir, commit_message=f'Epoch {epoch}',
                        ignore_patterns=['step_*', 'epoch_*'])
    accelerator.end_training()
