def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    if args.report_to == 'wandb' and args.hub_token is not None:
        raise ValueError(
            'You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token. Please use `huggingface-cli login` to authenticate with the Hub.'
            )
    accelerator_project_config = ProjectConfiguration(project_dir=args.
        output_dir, logging_dir=logging_dir)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(gradient_accumulation_steps=args.
        gradient_accumulation_steps, mixed_precision=args.mixed_precision,
        log_with=args.report_to, project_config=accelerator_project_config,
        kwargs_handlers=[kwargs])
    if args.report_to == 'tensorboard':
        if not is_tensorboard_available():
            raise ImportError(
                'Make sure to install tensorboard if you want to use it for logging during training.'
                )
    elif args.report_to == 'wandb':
        if not is_wandb_available():
            raise ImportError(
                'Make sure to install wandb if you want to use it for logging during training.'
                )
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
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.
                output_dir).name, exist_ok=True, token=args.hub_token).repo_id
    initial_discretization_steps = get_discretization_steps(0, args.
        max_train_steps, s_0=args.discretization_s_0, s_1=args.
        discretization_s_1, constant=args.constant_discretization_steps)
    noise_scheduler = CMStochasticIterativeScheduler(num_train_timesteps=
        initial_discretization_steps, sigma_min=args.sigma_min, sigma_max=
        args.sigma_max, rho=args.rho)
    if args.pretrained_model_name_or_path is not None:
        logger.info(
            f'Loading pretrained U-Net weights from {args.pretrained_model_name_or_path}... '
            )
        unet = UNet2DModel.from_pretrained(args.
            pretrained_model_name_or_path, subfolder='unet', revision=args.
            revision, variant=args.variant)
    elif args.model_config_name_or_path is None:
        if not args.class_conditional and (args.num_classes is not None or 
            args.class_embed_type is not None):
            logger.warning(
                f'`--class_conditional` is set to `False` but `--num_classes` is set to {args.num_classes} and `--class_embed_type` is set to {args.class_embed_type}. These values will be overridden to `None`.'
                )
            args.num_classes = None
            args.class_embed_type = None
        elif args.class_conditional and args.num_classes is None and args.class_embed_type is None:
            logger.warning(
                '`--class_conditional` is set to `True` but neither `--num_classes` nor `--class_embed_type` is set.`class_conditional` will be overridden to `False`.'
                )
            args.class_conditional = False
        unet = UNet2DModel(sample_size=args.resolution, in_channels=3,
            out_channels=3, layers_per_block=2, block_out_channels=(128, 
            128, 256, 256, 512, 512), down_block_types=('DownBlock2D',
            'DownBlock2D', 'DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D',
            'DownBlock2D'), up_block_types=('UpBlock2D', 'AttnUpBlock2D',
            'UpBlock2D', 'UpBlock2D', 'UpBlock2D', 'UpBlock2D'),
            class_embed_type=args.class_embed_type, num_class_embeds=args.
            num_classes)
    else:
        config = UNet2DModel.load_config(args.model_config_name_or_path)
        unet = UNet2DModel.from_config(config)
    unet.train()
    if args.use_ema:
        if args.ema_min_decay is None:
            args.ema_min_decay = args.ema_max_decay
        ema_unet = EMAModel(unet.parameters(), decay=args.ema_max_decay,
            min_decay=args.ema_min_decay, use_ema_warmup=args.
            use_ema_warmup, inv_gamma=args.ema_inv_gamma, power=args.
            ema_power, model_cls=UNet2DModel, model_config=unet.config)
    teacher_unet = UNet2DModel.from_config(unet.config)
    teacher_unet.load_state_dict(unet.state_dict())
    teacher_unet.train()
    teacher_unet.requires_grad_(False)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
    if args.cast_teacher:
        teacher_dtype = weight_dtype
    else:
        teacher_dtype = torch.float32
    teacher_unet.to(accelerator.device)
    if args.use_ema:
        ema_unet.to(accelerator.device)
    if version.parse(accelerate.__version__) >= version.parse('0.16.0'):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                teacher_unet.save_pretrained(os.path.join(output_dir,
                    'unet_teacher'))
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir,
                        'unet_ema'))
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, 'unet'))
                    weights.pop()

        def load_model_hook(models, input_dir):
            load_model = UNet2DModel.from_pretrained(os.path.join(input_dir,
                'unet_teacher'))
            teacher_unet.load_state_dict(load_model.state_dict())
            teacher_unet.to(accelerator.device)
            del load_model
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(
                    input_dir, 'unet_ema'), UNet2DModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
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
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse('0.0.16'):
                logger.warning(
                    'xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details.'
                    )
            unet.enable_xformers_memory_efficient_attention()
            teacher_unet.enable_xformers_memory_efficient_attention()
            if args.use_ema:
                ema_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                'xformers is not available. Make sure it is installed correctly'
                )
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if args.optimizer_type == 'radam':
        optimizer_class = torch.optim.RAdam
    elif args.optimizer_type == 'adamw':
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
    else:
        raise ValueError(
            f'Optimizer type {args.optimizer_type} is not supported. Currently supported optimizer types are `radam` and `adamw`.'
            )
    optimizer = optimizer_class(unet.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.
        adam_weight_decay, eps=args.adam_epsilon)
    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name, args.dataset_config_name,
            cache_dir=args.cache_dir, split='train')
    else:
        dataset = load_dataset('imagefolder', data_dir=args.train_data_dir,
            cache_dir=args.cache_dir, split='train')
    interpolation_mode = resolve_interpolation_mode(args.interpolation_type)
    augmentations = transforms.Compose([transforms.Resize(args.resolution,
        interpolation=interpolation_mode), transforms.CenterCrop(args.
        resolution) if args.center_crop else transforms.RandomCrop(args.
        resolution), transforms.RandomHorizontalFlip() if args.random_flip else
        transforms.Lambda(lambda x: x), transforms.ToTensor(), transforms.
        Normalize([0.5], [0.5])])

    def transform_images(examples):
        images = [augmentations(image.convert('RGB')) for image in examples
            [args.dataset_image_column_name]]
        batch_dict = {'images': images}
        if args.class_conditional:
            batch_dict['class_labels'] = examples[args.
                dataset_class_label_column_name]
        return batch_dict
    logger.info(f'Dataset size: {len(dataset)}')
    dataset.set_transform(transform_images)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args
        .train_batch_size, shuffle=True, num_workers=args.
        dataloader_num_workers)
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps, num_training_steps=args.
        max_train_steps)
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet,
        optimizer, train_dataloader, lr_scheduler)

    def recalculate_num_discretization_step_values(discretization_steps,
        skip_steps):
        """
        Recalculates all quantities depending on the number of discretization steps N.
        """
        noise_scheduler = CMStochasticIterativeScheduler(num_train_timesteps
            =discretization_steps, sigma_min=args.sigma_min, sigma_max=args
            .sigma_max, rho=args.rho)
        current_timesteps = get_karras_sigmas(discretization_steps, args.
            sigma_min, args.sigma_max, args.rho)
        valid_teacher_timesteps_plus_one = current_timesteps[:len(
            current_timesteps) - skip_steps + 1]
        timestep_weights = get_discretized_lognormal_weights(
            valid_teacher_timesteps_plus_one, p_mean=args.p_mean, p_std=
            args.p_std)
        timestep_loss_weights = get_loss_weighting_schedule(
            valid_teacher_timesteps_plus_one)
        current_timesteps = current_timesteps.to(accelerator.device)
        timestep_weights = timestep_weights.to(accelerator.device)
        timestep_loss_weights = timestep_loss_weights.to(accelerator.device)
        return (noise_scheduler, current_timesteps, timestep_weights,
            timestep_loss_weights)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
    args.num_train_epochs = math.ceil(args.max_train_steps /
        num_update_steps_per_epoch)
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=
            tracker_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    total_batch_size = (args.train_batch_size * accelerator.num_processes *
        args.gradient_accumulation_steps)
    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(dataset)}')
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
    if args.huber_c is None:
        args.huber_c = 0.00054 * args.resolution * math.sqrt(unet.config.
            in_channels)
    current_discretization_steps = get_discretization_steps(initial_global_step
        , args.max_train_steps, s_0=args.discretization_s_0, s_1=args.
        discretization_s_1, constant=args.constant_discretization_steps)
    current_skip_steps = get_skip_steps(initial_global_step, initial_skip=
        args.skip_steps)
    if current_skip_steps >= current_discretization_steps:
        raise ValueError(
            f'The current skip steps is {current_skip_steps}, but should be smaller than the current number of discretization steps {current_discretization_steps}'
            )
    (noise_scheduler, current_timesteps, timestep_weights,
        timestep_loss_weights) = (recalculate_num_discretization_step_values
        (current_discretization_steps, current_skip_steps))
    progress_bar = tqdm(range(0, args.max_train_steps), initial=
        initial_global_step, desc='Steps', disable=not accelerator.
        is_local_main_process)
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images'].to(weight_dtype)
            if args.class_conditional:
                class_labels = batch['class_labels']
            else:
                class_labels = None
            bsz = clean_images.shape[0]
            timestep_indices = torch.multinomial(timestep_weights, bsz,
                replacement=True).long()
            teacher_timesteps = current_timesteps[timestep_indices]
            student_timesteps = current_timesteps[timestep_indices +
                current_skip_steps]
            noise = torch.randn(clean_images.shape, dtype=weight_dtype,
                device=clean_images.device)
            teacher_noisy_images = add_noise(clean_images, noise,
                teacher_timesteps)
            student_noisy_images = add_noise(clean_images, noise,
                student_timesteps)
            teacher_rescaled_timesteps = get_noise_preconditioning(
                teacher_timesteps, args.noise_precond_type)
            student_rescaled_timesteps = get_noise_preconditioning(
                student_timesteps, args.noise_precond_type)
            c_in_teacher = get_input_preconditioning(teacher_timesteps,
                input_precond_type=args.input_precond_type)
            c_in_student = get_input_preconditioning(student_timesteps,
                input_precond_type=args.input_precond_type)
            c_skip_teacher, c_out_teacher = scalings_for_boundary_conditions(
                teacher_timesteps)
            c_skip_student, c_out_student = scalings_for_boundary_conditions(
                student_timesteps)
            c_skip_teacher, c_out_teacher, c_in_teacher = [append_dims(x,
                clean_images.ndim) for x in [c_skip_teacher, c_out_teacher,
                c_in_teacher]]
            c_skip_student, c_out_student, c_in_student = [append_dims(x,
                clean_images.ndim) for x in [c_skip_student, c_out_student,
                c_in_student]]
            with accelerator.accumulate(unet):
                dropout_state = torch.get_rng_state()
                student_model_output = unet(c_in_student *
                    student_noisy_images, student_rescaled_timesteps,
                    class_labels=class_labels).sample
                student_denoise_output = (c_skip_student *
                    student_noisy_images + c_out_student * student_model_output
                    )
                with torch.no_grad(), torch.autocast('cuda', dtype=
                    teacher_dtype):
                    torch.set_rng_state(dropout_state)
                    teacher_model_output = teacher_unet(c_in_teacher *
                        teacher_noisy_images, teacher_rescaled_timesteps,
                        class_labels=class_labels).sample
                    teacher_denoise_output = (c_skip_teacher *
                        teacher_noisy_images + c_out_teacher *
                        teacher_model_output)
                if args.prediction_type == 'sample':
                    lambda_t = _extract_into_tensor(timestep_loss_weights,
                        timestep_indices, (bsz,) + (1,) * (clean_images.
                        ndim - 1))
                    loss = lambda_t * (torch.sqrt((student_denoise_output.
                        float() - teacher_denoise_output.float()) ** 2 + 
                        args.huber_c ** 2) - args.huber_c)
                    loss = loss.mean()
                else:
                    raise ValueError(
                        f'Unsupported prediction type: {args.prediction_type}. Currently, only `sample` is supported.'
                        )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.
                        max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                teacher_unet.load_state_dict(unet.state_dict())
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    new_discretization_steps = get_discretization_steps(
                        global_step, args.max_train_steps, s_0=args.
                        discretization_s_0, s_1=args.discretization_s_1,
                        constant=args.constant_discretization_steps)
                    current_skip_steps = get_skip_steps(global_step,
                        initial_skip=args.skip_steps)
                    if current_skip_steps >= new_discretization_steps:
                        raise ValueError(
                            f'The current skip steps is {current_skip_steps}, but should be smaller than the current number of discretization steps {new_discretization_steps}.'
                            )
                    if (new_discretization_steps !=
                        current_discretization_steps):
                        (noise_scheduler, current_timesteps,
                            timestep_weights, timestep_loss_weights) = (
                            recalculate_num_discretization_step_values(
                            new_discretization_steps, current_skip_steps))
                        current_discretization_steps = new_discretization_steps
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
                        log_validation(unet, noise_scheduler, args,
                            accelerator, weight_dtype, global_step, 'teacher')
                        if args.use_ema:
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                            log_validation(unet, noise_scheduler, args,
                                accelerator, weight_dtype, global_step,
                                'ema_student')
                            ema_unet.restore(unet.parameters())
            logs = {'loss': loss.detach().item(), 'lr': lr_scheduler.
                get_last_lr()[0], 'step': global_step}
            if args.use_ema:
                logs['ema_decay'] = ema_unet.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        pipeline = ConsistencyModelPipeline(unet=unet, scheduler=
            noise_scheduler)
        pipeline.save_pretrained(args.output_dir)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
            unet.save_pretrained(os.path.join(args.output_dir, 'ema_unet'))
        if args.push_to_hub:
            upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message='End of training', ignore_patterns=['step_*',
                'epoch_*'])
    accelerator.end_training()
