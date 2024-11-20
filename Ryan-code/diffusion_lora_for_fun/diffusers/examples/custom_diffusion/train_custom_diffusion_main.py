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
    if accelerator.is_main_process:
        accelerator.init_trackers('custom-diffusion', config=vars(args))
    if args.seed is not None:
        set_seed(args.seed)
    if args.concepts_list is None:
        args.concepts_list = [{'instance_prompt': args.instance_prompt,
            'class_prompt': args.class_prompt, 'instance_data_dir': args.
            instance_data_dir, 'class_data_dir': args.class_data_dir}]
    else:
        with open(args.concepts_list, 'r') as f:
            args.concepts_list = json.load(f)
    if args.with_prior_preservation:
        for i, concept in enumerate(args.concepts_list):
            class_images_dir = Path(concept['class_data_dir'])
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True, exist_ok=True)
            if args.real_prior:
                assert (class_images_dir / 'images').exists(
                    ), f'Please run: python retrieve.py --class_prompt "{concept[\'class_prompt\']}" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}'
                assert len(list((class_images_dir / 'images').iterdir())
                    ) == args.num_class_images, f'Please run: python retrieve.py --class_prompt "{concept[\'class_prompt\']}" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}'
                assert (class_images_dir / 'caption.txt').exists(
                    ), f'Please run: python retrieve.py --class_prompt "{concept[\'class_prompt\']}" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}'
                assert (class_images_dir / 'images.txt').exists(
                    ), f'Please run: python retrieve.py --class_prompt "{concept[\'class_prompt\']}" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}'
                concept['class_prompt'] = os.path.join(class_images_dir,
                    'caption.txt')
                concept['class_data_dir'] = os.path.join(class_images_dir,
                    'images.txt')
                args.concepts_list[i] = concept
                accelerator.wait_for_everyone()
            else:
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
                        pretrained_model_name_or_path, torch_dtype=
                        torch_dtype, safety_checker=None, revision=args.
                        revision, variant=args.variant)
                    pipeline.set_progress_bar_config(disable=True)
                    num_new_images = args.num_class_images - cur_class_images
                    logger.info(
                        f'Number of class images to sample: {num_new_images}.')
                    sample_dataset = PromptDataset(concept['class_prompt'],
                        num_new_images)
                    sample_dataloader = torch.utils.data.DataLoader(
                        sample_dataset, batch_size=args.sample_batch_size)
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
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
        subfolder='vae', revision=args.revision, variant=args.variant)
    unet = UNet2DConditionModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='unet', revision=args.
        revision, variant=args.variant)
    modifier_token_id = []
    initializer_token_id = []
    if args.modifier_token is not None:
        args.modifier_token = args.modifier_token.split('+')
        args.initializer_token = args.initializer_token.split('+')
        if len(args.modifier_token) > len(args.initializer_token):
            raise ValueError(
                'You must specify + separated initializer token for each modifier token.'
                )
        for modifier_token, initializer_token in zip(args.modifier_token,
            args.initializer_token[:len(args.modifier_token)]):
            num_added_tokens = tokenizer.add_tokens(modifier_token)
            if num_added_tokens == 0:
                raise ValueError(
                    f'The tokenizer already contains the token {modifier_token}. Please pass a different `modifier_token` that is not already in the tokenizer.'
                    )
            token_ids = tokenizer.encode([initializer_token],
                add_special_tokens=False)
            print(token_ids)
            if len(token_ids) > 1:
                raise ValueError(
                    'The initializer token must be a single token.')
            initializer_token_id.append(token_ids[0])
            modifier_token_id.append(tokenizer.convert_tokens_to_ids(
                modifier_token))
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for x, y in zip(modifier_token_id, initializer_token_id):
            token_embeds[x] = token_embeds[y]
        params_to_freeze = itertools.chain(text_encoder.text_model.encoder.
            parameters(), text_encoder.text_model.final_layer_norm.
            parameters(), text_encoder.text_model.embeddings.
            position_embedding.parameters())
        freeze_params(params_to_freeze)
    vae.requires_grad_(False)
    if args.modifier_token is None:
        text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    if (accelerator.mixed_precision != 'fp16' and args.modifier_token is not
        None):
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    attention_class = CustomDiffusionAttnProcessor2_0 if hasattr(F,
        'scaled_dot_product_attention') else CustomDiffusionAttnProcessor
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse('0.0.16'):
                logger.warning(
                    'xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details.'
                    )
            attention_class = CustomDiffusionXFormersAttnProcessor
        else:
            raise ValueError(
                'xformers is not available. Make sure it is installed correctly'
                )
    train_kv = True
    train_q_out = False if args.freeze_model == 'crossattn_kv' else True
    custom_diffusion_attn_procs = {}
    st = unet.state_dict()
    for name, _ in unet.attn_processors.items():
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
        layer_name = name.split('.processor')[0]
        weights = {'to_k_custom_diffusion.weight': st[layer_name +
            '.to_k.weight'], 'to_v_custom_diffusion.weight': st[layer_name +
            '.to_v.weight']}
        if train_q_out:
            weights['to_q_custom_diffusion.weight'] = st[layer_name +
                '.to_q.weight']
            weights['to_out_custom_diffusion.0.weight'] = st[layer_name +
                '.to_out.0.weight']
            weights['to_out_custom_diffusion.0.bias'] = st[layer_name +
                '.to_out.0.bias']
        if cross_attention_dim is not None:
            custom_diffusion_attn_procs[name] = attention_class(train_kv=
                train_kv, train_q_out=train_q_out, hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim).to(unet.device)
            custom_diffusion_attn_procs[name].load_state_dict(weights)
        else:
            custom_diffusion_attn_procs[name] = attention_class(train_kv=
                False, train_q_out=False, hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim)
    del st
    unet.set_attn_processor(custom_diffusion_attn_procs)
    custom_diffusion_layers = AttnProcsLayers(unet.attn_processors)
    accelerator.register_for_checkpointing(custom_diffusion_layers)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.modifier_token is not None:
            text_encoder.gradient_checkpointing_enable()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.
            gradient_accumulation_steps * args.train_batch_size *
            accelerator.num_processes)
        if args.with_prior_preservation:
            args.learning_rate = args.learning_rate * 2.0
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
    optimizer = optimizer_class(itertools.chain(text_encoder.
        get_input_embeddings().parameters(), custom_diffusion_layers.
        parameters()) if args.modifier_token is not None else
        custom_diffusion_layers.parameters(), lr=args.learning_rate, betas=
        (args.adam_beta1, args.adam_beta2), weight_decay=args.
        adam_weight_decay, eps=args.adam_epsilon)
    train_dataset = CustomDiffusionDataset(concepts_list=args.concepts_list,
        tokenizer=tokenizer, with_prior_preservation=args.
        with_prior_preservation, size=args.resolution, mask_size=vae.encode
        (torch.randn(1, 3, args.resolution, args.resolution).to(dtype=
        weight_dtype).to(accelerator.device)).latent_dist.sample().size()[-
        1], center_crop=args.center_crop, num_class_images=args.
        num_class_images, hflip=args.hflip, aug=not args.noaug)
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
        num_training_steps=args.max_train_steps * accelerator.num_processes)
    if args.modifier_token is not None:
        (custom_diffusion_layers, text_encoder, optimizer, train_dataloader,
            lr_scheduler) = (accelerator.prepare(custom_diffusion_layers,
            text_encoder, optimizer, train_dataloader, lr_scheduler))
    else:
        (custom_diffusion_layers, optimizer, train_dataloader, lr_scheduler
            ) = (accelerator.prepare(custom_diffusion_layers, optimizer,
            train_dataloader, lr_scheduler))
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
    args.num_train_epochs = math.ceil(args.max_train_steps /
        num_update_steps_per_epoch)
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
        if args.modifier_token is not None:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet), accelerator.accumulate(
                text_encoder):
                latents = vae.encode(batch['pixel_values'].to(dtype=
                    weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.
                    num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise,
                    timesteps)
                encoder_hidden_states = text_encoder(batch['input_ids'])[0]
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
                if args.with_prior_preservation:
                    model_pred, model_pred_prior = torch.chunk(model_pred, 
                        2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    mask = torch.chunk(batch['mask'], 2, dim=0)[0]
                    loss = F.mse_loss(model_pred.float(), target.float(),
                        reduction='none')
                    loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])
                        ).mean()
                    prior_loss = F.mse_loss(model_pred_prior.float(),
                        target_prior.float(), reduction='mean')
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    mask = batch['mask']
                    loss = F.mse_loss(model_pred.float(), target.float(),
                        reduction='none')
                    loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])
                        ).mean()
                accelerator.backward(loss)
                if args.modifier_token is not None:
                    if accelerator.num_processes > 1:
                        grads_text_encoder = (text_encoder.module.
                            get_input_embeddings().weight.grad)
                    else:
                        grads_text_encoder = text_encoder.get_input_embeddings(
                            ).weight.grad
                    index_grads_to_zero = torch.arange(len(tokenizer)
                        ) != modifier_token_id[0]
                    for i in range(1, len(modifier_token_id)):
                        index_grads_to_zero = index_grads_to_zero & (torch.
                            arange(len(tokenizer)) != modifier_token_id[i])
                    grads_text_encoder.data[index_grads_to_zero, :
                        ] = grads_text_encoder.data[index_grads_to_zero, :
                        ].fill_(0)
                if accelerator.sync_gradients:
                    params_to_clip = (itertools.chain(text_encoder.
                        parameters(), custom_diffusion_layers.parameters()) if
                        args.modifier_token is not None else
                        custom_diffusion_layers.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.
                        max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
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
            logs = {'loss': loss.detach().item(), 'lr': lr_scheduler.
                get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
            if accelerator.is_main_process:
                images = []
                if (args.validation_prompt is not None and global_step %
                    args.validation_steps == 0):
                    logger.info(
                        f"""Running validation... 
 Generating {args.num_validation_images} images with prompt: {args.validation_prompt}."""
                        )
                    pipeline = DiffusionPipeline.from_pretrained(args.
                        pretrained_model_name_or_path, unet=accelerator.
                        unwrap_model(unet), text_encoder=accelerator.
                        unwrap_model(text_encoder), tokenizer=tokenizer,
                        revision=args.revision, variant=args.variant,
                        torch_dtype=weight_dtype)
                    pipeline.scheduler = (DPMSolverMultistepScheduler.
                        from_config(pipeline.scheduler.config))
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)
                    generator = torch.Generator(device=accelerator.device
                        ).manual_seed(args.seed)
                    images = [pipeline(args.validation_prompt,
                        num_inference_steps=25, generator=generator, eta=
                        1.0).images[0] for _ in range(args.
                        num_validation_images)]
                    for tracker in accelerator.trackers:
                        if tracker.name == 'tensorboard':
                            np_images = np.stack([np.asarray(img) for img in
                                images])
                            tracker.writer.add_images('validation',
                                np_images, epoch, dataformats='NHWC')
                        if tracker.name == 'wandb':
                            tracker.log({'validation': [wandb.Image(image,
                                caption=f'{i}: {args.validation_prompt}') for
                                i, image in enumerate(images)]})
                    del pipeline
                    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unet.save_attn_procs(args.output_dir, safe_serialization=not args.
            no_safe_serialization)
        save_new_embed(text_encoder, modifier_token_id, accelerator, args,
            args.output_dir, safe_serialization=not args.no_safe_serialization)
        pipeline = DiffusionPipeline.from_pretrained(args.
            pretrained_model_name_or_path, revision=args.revision, variant=
            args.variant, torch_dtype=weight_dtype)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline
            .scheduler.config)
        pipeline = pipeline.to(accelerator.device)
        weight_name = ('pytorch_custom_diffusion_weights.safetensors' if 
            not args.no_safe_serialization else
            'pytorch_custom_diffusion_weights.bin')
        pipeline.unet.load_attn_procs(args.output_dir, weight_name=weight_name)
        for token in args.modifier_token:
            token_weight_name = (f'{token}.safetensors' if not args.
                no_safe_serialization else f'{token}.bin')
            pipeline.load_textual_inversion(args.output_dir, weight_name=
                token_weight_name)
        if args.validation_prompt and args.num_validation_images > 0:
            generator = torch.Generator(device=accelerator.device).manual_seed(
                args.seed) if args.seed else None
            images = [pipeline(args.validation_prompt, num_inference_steps=
                25, generator=generator, eta=1.0).images[0] for _ in range(
                args.num_validation_images)]
            for tracker in accelerator.trackers:
                if tracker.name == 'tensorboard':
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images('test', np_images, epoch,
                        dataformats='NHWC')
                if tracker.name == 'wandb':
                    tracker.log({'test': [wandb.Image(image, caption=
                        f'{i}: {args.validation_prompt}') for i, image in
                        enumerate(images)]})
        if args.push_to_hub:
            save_model_card(repo_id, images=images, base_model=args.
                pretrained_model_name_or_path, prompt=args.instance_prompt,
                repo_folder=args.output_dir)
            api = HfApi(token=args.hub_token)
            api.upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message='End of training', ignore_patterns=['step_*',
                'epoch_*'])
    accelerator.end_training()
