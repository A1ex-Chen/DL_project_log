def main():
    args = parse_args()
    if args.report_to == 'wandb' and args.hub_token is not None:
        raise ValueError(
            'You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token. Please use `huggingface-cli login` to authenticate with the Hub.'
            )
    if args.non_ema_revision is not None:
        deprecate('non_ema_revision!=None', '0.15.0', message=
            "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to use `--variant=non_ema` instead."
            )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    if torch.backends.mps.is_available() and args.mixed_precision == 'bf16':
        raise ValueError(
            'Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead.'
            )
    accelerator_project_config = ProjectConfiguration(project_dir=args.
        output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(gradient_accumulation_steps=args.
        gradient_accumulation_steps, mixed_precision=args.mixed_precision,
        log_with=args.report_to, project_config=accelerator_project_config)
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    generator = torch.Generator(device=accelerator.device).manual_seed(args
        .seed)
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
    vae_path = (args.pretrained_model_name_or_path if args.
        pretrained_vae_model_name_or_path is None else args.
        pretrained_vae_model_name_or_path)
    vae = AutoencoderKL.from_pretrained(vae_path, subfolder='vae' if args.
        pretrained_vae_model_name_or_path is None else None, revision=args.
        revision, variant=args.variant)
    unet = UNet2DConditionModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='unet', revision=args.
        revision, variant=args.variant)
    logger.info(
        'Initializing the XL InstructPix2Pix UNet from the pretrained UNet.')
    in_channels = 8
    out_channels = unet.conv_in.out_channels
    unet.register_to_config(in_channels=in_channels)
    with torch.no_grad():
        new_conv_in = nn.Conv2d(in_channels, out_channels, unet.conv_in.
            kernel_size, unet.conv_in.stride, unet.conv_in.padding)
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        unet.conv_in = new_conv_in
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=
            UNet2DConditionModel, model_config=unet.config)
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

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    if version.parse(accelerate.__version__) >= version.parse('0.16.0'):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir,
                        'unet_ema'))
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, 'unet'))
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(
                    input_dir, 'unet_ema'), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model
            for i in range(len(models)):
                model = models.pop()
                load_model = UNet2DConditionModel.from_pretrained(input_dir,
                    subfolder='unet')
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
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
                'Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`'
                )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(unet.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.
        adam_weight_decay, eps=args.adam_epsilon)
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
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.original_image_column is None:
        original_image_column = dataset_columns[0
            ] if dataset_columns is not None else column_names[0]
    else:
        original_image_column = args.original_image_column
        if original_image_column not in column_names:
            raise ValueError(
                f"--original_image_column' value '{args.original_image_column}' needs to be one of: {', '.join(column_names)}"
                )
    if args.edit_prompt_column is None:
        edit_prompt_column = dataset_columns[1
            ] if dataset_columns is not None else column_names[1]
    else:
        edit_prompt_column = args.edit_prompt_column
        if edit_prompt_column not in column_names:
            raise ValueError(
                f"--edit_prompt_column' value '{args.edit_prompt_column}' needs to be one of: {', '.join(column_names)}"
                )
    if args.edited_image_column is None:
        edited_image_column = dataset_columns[2
            ] if dataset_columns is not None else column_names[2]
    else:
        edited_image_column = args.edited_image_column
        if edited_image_column not in column_names:
            raise ValueError(
                f"--edited_image_column' value '{args.edited_image_column}' needs to be one of: {', '.join(column_names)}"
                )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
        warnings.warn(
            f'weight_dtype {weight_dtype} may cause nan during vae encoding',
            UserWarning)
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
        warnings.warn(
            f'weight_dtype {weight_dtype} may cause nan during vae encoding',
            UserWarning)

    def tokenize_captions(captions, tokenizer):
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length,
            padding='max_length', truncation=True, return_tensors='pt')
        return inputs.input_ids
    train_transforms = transforms.Compose([transforms.CenterCrop(args.
        resolution) if args.center_crop else transforms.RandomCrop(args.
        resolution), transforms.RandomHorizontalFlip() if args.random_flip else
        transforms.Lambda(lambda x: x)])

    def preprocess_images(examples):
        original_images = np.concatenate([convert_to_np(image, args.
            resolution) for image in examples[original_image_column]])
        edited_images = np.concatenate([convert_to_np(image, args.
            resolution) for image in examples[edited_image_column]])
        images = np.concatenate([original_images, edited_images])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        return train_transforms(images)
    tokenizer_1 = AutoTokenizer.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='tokenizer', revision=args
        .revision, use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='tokenizer_2', revision=
        args.revision, use_fast=False)
    text_encoder_cls_1 = import_model_class_from_model_name_or_path(args.
        pretrained_model_name_or_path, args.revision)
    text_encoder_cls_2 = import_model_class_from_model_name_or_path(args.
        pretrained_model_name_or_path, args.revision, subfolder=
        'text_encoder_2')
    noise_scheduler = DDPMScheduler.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='scheduler')
    text_encoder_1 = text_encoder_cls_1.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='text_encoder', revision=
        args.revision, variant=args.variant)
    text_encoder_2 = text_encoder_cls_2.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='text_encoder_2', revision
        =args.revision, variant=args.variant)
    text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    tokenizers = [tokenizer_1, tokenizer_2]
    text_encoders = [text_encoder_1, text_encoder_2]
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.train()

    def encode_prompt(text_encoders, tokenizers, prompt):
        prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(prompt, padding='max_length',
                max_length=tokenizer.model_max_length, truncation=True,
                return_tensors='pt')
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding='longest',
                return_tensors='pt').input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1
                ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, 
                    tokenizer.model_max_length - 1:-1])
                logger.warning(
                    f'The following part of your input was truncated because CLIP can only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}'
                    )
            prompt_embeds = text_encoder(text_input_ids.to(text_encoder.
                device), output_hidden_states=True)
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds

    def encode_prompts(text_encoders, tokenizers, prompts):
        prompt_embeds_all = []
        pooled_prompt_embeds_all = []
        for prompt in prompts:
            prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders,
                tokenizers, prompt)
            prompt_embeds_all.append(prompt_embeds)
            pooled_prompt_embeds_all.append(pooled_prompt_embeds)
        return torch.stack(prompt_embeds_all), torch.stack(
            pooled_prompt_embeds_all)

    def compute_embeddings_for_prompts(prompts, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds_all, pooled_prompt_embeds_all = encode_prompts(
                text_encoders, tokenizers, prompts)
            add_text_embeds_all = pooled_prompt_embeds_all
            prompt_embeds_all = prompt_embeds_all.to(accelerator.device)
            add_text_embeds_all = add_text_embeds_all.to(accelerator.device)
        return prompt_embeds_all, add_text_embeds_all

    def compute_null_conditioning():
        null_conditioning_list = []
        for a_tokenizer, a_text_encoder in zip(tokenizers, text_encoders):
            null_conditioning_list.append(a_text_encoder(tokenize_captions(
                [''], tokenizer=a_tokenizer).to(accelerator.device),
                output_hidden_states=True).hidden_states[-2])
        return torch.concat(null_conditioning_list, dim=-1)
    null_conditioning = compute_null_conditioning()

    def compute_time_ids():
        crops_coords_top_left = (args.crops_coords_top_left_h, args.
            crops_coords_top_left_w)
        original_size = target_size = args.resolution, args.resolution
        add_time_ids = list(original_size + crops_coords_top_left + target_size
            )
        add_time_ids = torch.tensor([add_time_ids], dtype=weight_dtype)
        return add_time_ids.to(accelerator.device).repeat(args.
            train_batch_size, 1)
    add_time_ids = compute_time_ids()

    def preprocess_train(examples):
        preprocessed_images = preprocess_images(examples)
        original_images, edited_images = preprocessed_images.chunk(2)
        original_images = original_images.reshape(-1, 3, args.resolution,
            args.resolution)
        edited_images = edited_images.reshape(-1, 3, args.resolution, args.
            resolution)
        examples['original_pixel_values'] = original_images
        examples['edited_pixel_values'] = edited_images
        captions = list(examples[edit_prompt_column])
        prompt_embeds_all, add_text_embeds_all = (
            compute_embeddings_for_prompts(captions, text_encoders, tokenizers)
            )
        examples['prompt_embeds'] = prompt_embeds_all
        examples['add_text_embeds'] = add_text_embeds_all
        return examples
    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset['train'] = dataset['train'].shuffle(seed=args.seed).select(
                range(args.max_train_samples))
        train_dataset = dataset['train'].with_transform(preprocess_train)

    def collate_fn(examples):
        original_pixel_values = torch.stack([example[
            'original_pixel_values'] for example in examples])
        original_pixel_values = original_pixel_values.to(memory_format=
            torch.contiguous_format).float()
        edited_pixel_values = torch.stack([example['edited_pixel_values'] for
            example in examples])
        edited_pixel_values = edited_pixel_values.to(memory_format=torch.
            contiguous_format).float()
        prompt_embeds = torch.concat([example['prompt_embeds'] for example in
            examples], dim=0)
        add_text_embeds = torch.concat([example['add_text_embeds'] for
            example in examples], dim=0)
        return {'original_pixel_values': original_pixel_values,
            'edited_pixel_values': edited_pixel_values, 'prompt_embeds':
            prompt_embeds, 'add_text_embeds': add_text_embeds}
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
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet,
        optimizer, train_dataloader, lr_scheduler)
    if args.use_ema:
        ema_unet.to(accelerator.device)
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    else:
        vae.to(accelerator.device, dtype=TORCH_DTYPE_MAPPING[args.
            vae_precision])
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
    args.num_train_epochs = math.ceil(args.max_train_steps /
        num_update_steps_per_epoch)
    if accelerator.is_main_process:
        accelerator.init_trackers('instruct-pix2pix-xl', config=vars(args))
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
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                if args.pretrained_vae_model_name_or_path is not None:
                    edited_pixel_values = batch['edited_pixel_values'].to(dtype
                        =weight_dtype)
                else:
                    edited_pixel_values = batch['edited_pixel_values']
                latents = vae.encode(edited_pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    latents = latents.to(weight_dtype)
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.
                    num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise,
                    timesteps)
                encoder_hidden_states = batch['prompt_embeds']
                add_text_embeds = batch['add_text_embeds']
                if args.pretrained_vae_model_name_or_path is not None:
                    original_pixel_values = batch['original_pixel_values'].to(
                        dtype=weight_dtype)
                else:
                    original_pixel_values = batch['original_pixel_values']
                original_image_embeds = vae.encode(original_pixel_values
                    ).latent_dist.sample()
                if args.pretrained_vae_model_name_or_path is None:
                    original_image_embeds = original_image_embeds.to(
                        weight_dtype)
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device,
                        generator=generator)
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    encoder_hidden_states = torch.where(prompt_mask,
                        null_conditioning, encoder_hidden_states)
                    image_mask_dtype = original_image_embeds.dtype
                    image_mask = 1 - (random_p >= args.
                        conditioning_dropout_prob).to(image_mask_dtype) * (
                        random_p < 3 * args.conditioning_dropout_prob).to(
                        image_mask_dtype)
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    original_image_embeds = image_mask * original_image_embeds
                concatenated_noisy_latents = torch.cat([noisy_latents,
                    original_image_embeds], dim=1)
                if noise_scheduler.config.prediction_type == 'epsilon':
                    target = noise
                elif noise_scheduler.config.prediction_type == 'v_prediction':
                    target = noise_scheduler.get_velocity(latents, noise,
                        timesteps)
                else:
                    raise ValueError(
                        f'Unknown prediction type {noise_scheduler.config.prediction_type}'
                        )
                added_cond_kwargs = {'text_embeds': add_text_embeds,
                    'time_ids': add_time_ids}
                model_pred = unet(concatenated_noisy_latents, timesteps,
                    encoder_hidden_states, added_cond_kwargs=
                    added_cond_kwargs, return_dict=False)[0]
                loss = F.mse_loss(model_pred.float(), target.float(),
                    reduction='mean')
                avg_loss = accelerator.gather(loss.repeat(args.
                    train_batch_size)).mean()
                train_loss += avg_loss.item(
                    ) / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.
                        max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
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
            if global_step % args.validation_steps == 0:
                if (args.val_image_url_or_path is not None and args.
                    validation_prompt is not None):
                    if args.use_ema:
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())
                    pipeline = (StableDiffusionXLInstructPix2PixPipeline.
                        from_pretrained(args.pretrained_model_name_or_path,
                        unet=unwrap_model(unet), text_encoder=
                        text_encoder_1, text_encoder_2=text_encoder_2,
                        tokenizer=tokenizer_1, tokenizer_2=tokenizer_2, vae
                        =vae, revision=args.revision, variant=args.variant,
                        torch_dtype=weight_dtype))
                    log_validation(pipeline, args, accelerator, generator,
                        global_step, is_final_validation=False)
                    if args.use_ema:
                        ema_unet.restore(unet.parameters())
                    del pipeline
                    torch.cuda.empty_cache()
            if global_step >= args.max_train_steps:
                break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
        pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path, text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2, tokenizer=tokenizer_1,
            tokenizer_2=tokenizer_2, vae=vae, unet=unwrap_model(unet),
            revision=args.revision, variant=args.variant)
        pipeline.save_pretrained(args.output_dir)
        if args.push_to_hub:
            upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message='End of training', ignore_patterns=['step_*',
                'epoch_*'])
        if (args.val_image_url_or_path is not None and args.
            validation_prompt is not None):
            log_validation(pipeline, args, accelerator, generator,
                global_step, is_final_validation=True)
    accelerator.end_training()
