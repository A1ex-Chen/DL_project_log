def main(args):
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
    if (args.train_text_encoder and args.gradient_accumulation_steps > 1 and
        accelerator.num_processes > 1):
        raise ValueError(
            'Gradient accumulation is not supported when training the text encoder in distributed training. Please set gradient_accumulation_steps to 1. This feature will be supported in the future.'
            )
    instance_data_dir = []
    instance_prompt = []
    class_data_dir = [] if args.with_prior_preservation else None
    class_prompt = [] if args.with_prior_preservation else None
    if args.concepts_list:
        with open(args.concepts_list, 'r') as f:
            concepts_list = json.load(f)
        if args.validation_steps:
            args.validation_prompt = []
            args.validation_number_images = []
            args.validation_negative_prompt = []
            args.validation_inference_steps = []
            args.validation_guidance_scale = []
        for concept in concepts_list:
            instance_data_dir.append(concept['instance_data_dir'])
            instance_prompt.append(concept['instance_prompt'])
            if args.with_prior_preservation:
                try:
                    class_data_dir.append(concept['class_data_dir'])
                    class_prompt.append(concept['class_prompt'])
                except KeyError:
                    raise KeyError(
                        '`class_data_dir` or `class_prompt` not found in concepts_list while using `with_prior_preservation`.'
                        )
            else:
                if 'class_data_dir' in concept:
                    warnings.warn(
                        'Ignoring `class_data_dir` key, to use it you need to enable `with_prior_preservation`.'
                        )
                if 'class_prompt' in concept:
                    warnings.warn(
                        'Ignoring `class_prompt` key, to use it you need to enable `with_prior_preservation`.'
                        )
            if args.validation_steps:
                args.validation_prompt.append(concept.get(
                    'validation_prompt', None))
                args.validation_number_images.append(concept.get(
                    'validation_number_images', 4))
                args.validation_negative_prompt.append(concept.get(
                    'validation_negative_prompt', None))
                args.validation_inference_steps.append(concept.get(
                    'validation_inference_steps', 25))
                args.validation_guidance_scale.append(concept.get(
                    'validation_guidance_scale', 7.5))
    else:
        instance_data_dir = args.instance_data_dir.split(',')
        instance_prompt = args.instance_prompt.split(',')
        assert all(x == len(instance_data_dir) for x in [len(
            instance_data_dir), len(instance_prompt)]
            ), 'Instance data dir and prompt inputs are not of the same length.'
        if args.with_prior_preservation:
            class_data_dir = args.class_data_dir.split(',')
            class_prompt = args.class_prompt.split(',')
            assert all(x == len(instance_data_dir) for x in [len(
                instance_data_dir), len(instance_prompt), len(
                class_data_dir), len(class_prompt)]
                ), 'Instance & class data dir or prompt inputs are not of the same length.'
        if args.validation_steps:
            validation_prompts = args.validation_prompt.split(',')
            num_of_validation_prompts = len(validation_prompts)
            args.validation_prompt = validation_prompts
            args.validation_number_images = [args.validation_number_images
                ] * num_of_validation_prompts
            negative_validation_prompts = [None] * num_of_validation_prompts
            if args.validation_negative_prompt:
                negative_validation_prompts = (args.
                    validation_negative_prompt.split(','))
                while len(negative_validation_prompts
                    ) < num_of_validation_prompts:
                    negative_validation_prompts.append(None)
            args.validation_negative_prompt = negative_validation_prompts
            assert num_of_validation_prompts == len(negative_validation_prompts
                ), 'The length of negative prompts for validation is greater than the number of validation prompts.'
            args.validation_inference_steps = [args.validation_inference_steps
                ] * num_of_validation_prompts
            args.validation_guidance_scale = [args.validation_guidance_scale
                ] * num_of_validation_prompts
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
    if args.with_prior_preservation:
        for i in range(len(class_data_dir)):
            class_images_dir = Path(class_data_dir[i])
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
                    safety_checker=None, revision=args.revision)
                pipeline.set_progress_bar_config(disable=True)
                num_new_images = args.num_class_images - cur_class_images
                logger.info(
                    f'Number of class images to sample: {num_new_images}.')
                sample_dataset = PromptDataset(class_prompt[i], num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset,
                    batch_size=args.sample_batch_size)
                sample_dataloader = accelerator.prepare(sample_dataloader)
                pipeline.to(accelerator.device)
                for example in tqdm(sample_dataloader, desc=
                    'Generating class images', disable=not accelerator.
                    is_local_main_process):
                    images = pipeline(example['prompt']).images
                    for ii, image in enumerate(images):
                        hash_image = insecure_hashlib.sha1(image.tobytes()
                            ).hexdigest()
                        image_filename = (class_images_dir /
                            f"{example['index'][ii] + cur_class_images}-{hash_image}.jpg"
                            )
                        image.save(image_filename)
                del pipeline
                del sample_dataloader
                del sample_dataset
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    if accelerator.is_main_process:
        if args.output_dir is not None:
            makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.
                output_dir).name, exist_ok=True, token=args.hub_token).repo_id
    tokenizer = None
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
        args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
        subfolder='vae', revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='unet', revision=args.revision
        )
    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                'xformers is not available. Make sure it is installed correctly'
                )
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()
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
    train_dataset = DreamBoothDataset(instance_data_root=instance_data_dir,
        instance_prompt=instance_prompt, class_data_root=class_data_dir,
        class_prompt=class_prompt, tokenizer=tokenizer, size=args.
        resolution, center_crop=args.center_crop)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.train_batch_size, shuffle=True, collate_fn=lambda
        examples: collate_fn(len(instance_data_dir), examples, args.
        with_prior_preservation), num_workers=1)
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
            path = basename(args.resume_from_checkpoint)
        else:
            dirs = listdir(args.output_dir)
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
            accelerator.load_state(join(args.output_dir, path))
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
        if args.train_text_encoder:
            text_encoder.train()
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
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                time_steps = torch.randint(0, noise_scheduler.config.
                    num_train_timesteps, (bsz,), device=latents.device)
                time_steps = time_steps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise,
                    time_steps)
                encoder_hidden_states = text_encoder(batch['input_ids'])[0]
                model_pred = unet(noisy_latents, time_steps,
                    encoder_hidden_states).sample
                if noise_scheduler.config.prediction_type == 'epsilon':
                    target = noise
                elif noise_scheduler.config.prediction_type == 'v_prediction':
                    target = noise_scheduler.get_velocity(latents, noise,
                        time_steps)
                else:
                    raise ValueError(
                        f'Unknown prediction type {noise_scheduler.config.prediction_type}'
                        )
                if args.with_prior_preservation:
                    model_pred, model_pred_prior = torch.chunk(model_pred, 
                        2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    loss = F.mse_loss(model_pred.float(), target.float(),
                        reduction='mean')
                    prior_loss = F.mse_loss(model_pred_prior.float(),
                        target_prior.float(), reduction='mean')
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(),
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
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = join(args.output_dir,
                            f'checkpoint-{global_step}')
                        accelerator.save_state(save_path)
                        logger.info(f'Saved state to {save_path}')
                    if args.validation_steps and any(args.validation_prompt
                        ) and global_step % args.validation_steps == 0:
                        images_set = generate_validation_images(text_encoder,
                            tokenizer, unet, vae, args, accelerator,
                            weight_dtype)
                        for images, validation_prompt in zip(images_set,
                            args.validation_prompt):
                            if len(images) > 0:
                                label = str(uuid.uuid1())[:8]
                                log_validation_images_to_tracker(images,
                                    label, validation_prompt, accelerator,
                                    global_step)
            logs = {'loss': loss.detach().item(), 'lr': lr_scheduler.
                get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = DiffusionPipeline.from_pretrained(args.
            pretrained_model_name_or_path, unet=accelerator.unwrap_model(
            unet), text_encoder=accelerator.unwrap_model(text_encoder),
            revision=args.revision)
        pipeline.save_pretrained(args.output_dir)
        if args.push_to_hub:
            upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message='End of training', ignore_patterns=['step_*',
                'epoch_*'])
    accelerator.end_training()
