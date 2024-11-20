def main(args):
    if args.report_to == 'wandb' and args.hub_token is not None:
        raise ValueError(
            'You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token. Please use `huggingface-cli login` to authenticate with the Hub.'
            )
    if args.do_edm_style_training and args.snr_gamma is not None:
        raise ValueError(
            'Min-SNR formulation is not supported when conducting EDM-style training.'
            )
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.
        output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps=args.
        gradient_accumulation_steps, mixed_precision=args.mixed_precision,
        log_with=args.report_to, project_config=accelerator_project_config,
        kwargs_handlers=[kwargs])
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
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
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
            pipeline = StableDiffusionXLPipeline.from_pretrained(args.
                pretrained_model_name_or_path, torch_dtype=torch_dtype,
                revision=args.revision, variant=args.variant)
            pipeline.set_progress_bar_config(disable=True)
            num_new_images = args.num_class_images - cur_class_images
            logger.info(f'Number of class images to sample: {num_new_images}.')
            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset,
                batch_size=args.sample_batch_size)
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
        model_id = args.hub_model_id or Path(args.output_dir).name
        repo_id = None
        if args.push_to_hub:
            repo_id = create_repo(repo_id=model_id, exist_ok=True, token=
                args.hub_token).repo_id
    tokenizer_one = AutoTokenizer.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='tokenizer', revision=args
        .revision, variant=args.variant, use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='tokenizer_2', revision=
        args.revision, variant=args.variant, use_fast=False)
    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.
        pretrained_model_name_or_path, args.revision)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(args.
        pretrained_model_name_or_path, args.revision, subfolder=
        'text_encoder_2')
    scheduler_type = determine_scheduler_type(args.
        pretrained_model_name_or_path, args.revision)
    if 'EDM' in scheduler_type:
        args.do_edm_style_training = True
        noise_scheduler = EDMEulerScheduler.from_pretrained(args.
            pretrained_model_name_or_path, subfolder='scheduler')
        logger.info('Performing EDM-style training!')
    elif args.do_edm_style_training:
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.
            pretrained_model_name_or_path, subfolder='scheduler')
        logger.info('Performing EDM-style training!')
    else:
        noise_scheduler = DDPMScheduler.from_pretrained(args.
            pretrained_model_name_or_path, subfolder='scheduler')
    text_encoder_one = text_encoder_cls_one.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='text_encoder', revision=
        args.revision, variant=args.variant)
    text_encoder_two = text_encoder_cls_two.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='text_encoder_2', revision
        =args.revision, variant=args.variant)
    vae_path = (args.pretrained_model_name_or_path if args.
        pretrained_vae_model_name_or_path is None else args.
        pretrained_vae_model_name_or_path)
    vae = AutoencoderKL.from_pretrained(vae_path, subfolder='vae' if args.
        pretrained_vae_model_name_or_path is None else None, revision=args.
        revision, variant=args.variant)
    latents_mean = latents_std = None
    if hasattr(vae.config, 'latents_mean'
        ) and vae.config.latents_mean is not None:
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1)
    if hasattr(vae.config, 'latents_std'
        ) and vae.config.latents_std is not None:
        latents_std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1)
    unet = UNet2DConditionModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='unet', revision=args.
        revision, variant=args.variant)
    if args.train_text_encoder_ti:
        token_abstraction_list = ''.join(args.token_abstraction.split()).split(
            ',')
        logger.info(f'list of token identifiers: {token_abstraction_list}')
        token_abstraction_dict = {}
        token_idx = 0
        for i, token in enumerate(token_abstraction_list):
            token_abstraction_dict[token] = [f'<s{token_idx + i + j}>' for
                j in range(args.num_new_tokens_per_abstraction)]
            token_idx += args.num_new_tokens_per_abstraction - 1
        for token_abs, token_replacement in token_abstraction_dict.items():
            args.instance_prompt = args.instance_prompt.replace(token_abs,
                ''.join(token_replacement))
            if args.with_prior_preservation:
                args.class_prompt = args.class_prompt.replace(token_abs, ''
                    .join(token_replacement))
            if args.validation_prompt:
                args.validation_prompt = args.validation_prompt.replace(
                    token_abs, ''.join(token_replacement))
                print('validation prompt:', args.validation_prompt)
        embedding_handler = TokenEmbeddingsHandler([text_encoder_one,
            text_encoder_two], [tokenizer_one, tokenizer_two])
        inserting_toks = []
        for new_tok in token_abstraction_dict.values():
            inserting_toks.extend(new_tok)
        embedding_handler.initialize_new_tokens(inserting_toks=inserting_toks)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
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
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()
    if args.use_blora:
        target_modules = get_unet_lora_target_modules(unet, use_blora=True)
    elif args.lora_unet_blocks:
        target_blocks_list = ''.join(args.lora_unet_blocks.split()).split(',')
        logger.info(f'list of unet blocks to train: {target_blocks_list}')
        target_modules = get_unet_lora_target_modules(unet, use_blora=False,
            target_blocks=target_blocks_list)
    else:
        target_modules = ['to_k', 'to_q', 'to_v', 'to_out.0']
    unet_lora_config = LoraConfig(r=args.rank, use_dora=args.use_dora,
        lora_alpha=args.rank, init_lora_weights='gaussian', target_modules=
        target_modules)
    unet.add_adapter(unet_lora_config)
    if args.train_text_encoder:
        text_lora_config = LoraConfig(r=args.rank, use_dora=args.use_dora,
            lora_alpha=args.rank, init_lora_weights='gaussian',
            target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj'])
        text_encoder_one.add_adapter(text_lora_config)
        text_encoder_two.add_adapter(text_lora_config)
    elif args.train_text_encoder_ti:
        text_lora_parameters_one = []
        for name, param in text_encoder_one.named_parameters():
            if 'token_embedding' in name:
                param.data = param.to(dtype=torch.float32)
                param.requires_grad = True
                text_lora_parameters_one.append(param)
            else:
                param.requires_grad = False
        text_lora_parameters_two = []
        for name, param in text_encoder_two.named_parameters():
            if 'token_embedding' in name:
                param.data = param.to(dtype=torch.float32)
                param.requires_grad = True
                text_lora_parameters_two.append(param)
            else:
                param.requires_grad = False

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None
            for model in models:
                if isinstance(model, type(unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model))
                elif isinstance(model, type(unwrap_model(text_encoder_one))):
                    if args.train_text_encoder:
                        text_encoder_one_lora_layers_to_save = (
                            convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)))
                elif isinstance(model, type(unwrap_model(text_encoder_two))):
                    if args.train_text_encoder:
                        text_encoder_two_lora_layers_to_save = (
                            convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)))
                else:
                    raise ValueError(
                        f'unexpected save model: {model.__class__}')
                weights.pop()
            StableDiffusionXLPipeline.save_lora_weights(output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=
                text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save
                )
        if args.train_text_encoder_ti:
            embedding_handler.save_embeddings(
                f'{output_dir}/{args.output_dir}_emb.safetensors')

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_two))):
                text_encoder_two_ = model
            else:
                raise ValueError(f'unexpected save model: {model.__class__}')
        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(
            input_dir)
        unet_state_dict = {f"{k.replace('unet.', '')}": v for k, v in
            lora_state_dict.items() if k.startswith('unet.')}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_,
            unet_state_dict, adapter_name='default')
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, 'unexpected_keys',
                None)
            if unexpected_keys:
                logger.warning(
                    f'Loading adapter weights from state_dict led to unexpected keys not found in the model:  {unexpected_keys}. '
                    )
        if args.train_text_encoder:
            _set_state_dict_into_text_encoder(lora_state_dict, prefix=
                'text_encoder.', text_encoder=text_encoder_one_)
            _set_state_dict_into_text_encoder(lora_state_dict, prefix=
                'text_encoder_2.', text_encoder=text_encoder_two_)
        if args.mixed_precision == 'fp16':
            models = [unet_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_, text_encoder_two_])
            cast_training_params(models)
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.
            gradient_accumulation_steps * args.train_batch_size *
            accelerator.num_processes)
    if args.mixed_precision == 'fp16':
        models = [unet]
        if args.train_text_encoder:
            models.extend([text_encoder_one, text_encoder_two])
        cast_training_params(models, dtype=torch.float32)
    unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.
        parameters()))
    if args.train_text_encoder:
        text_lora_parameters_one = list(filter(lambda p: p.requires_grad,
            text_encoder_one.parameters()))
        text_lora_parameters_two = list(filter(lambda p: p.requires_grad,
            text_encoder_two.parameters()))
    freeze_text_encoder = not (args.train_text_encoder or args.
        train_text_encoder_ti)
    unet_lora_parameters_with_lr = {'params': unet_lora_parameters, 'lr':
        args.learning_rate}
    if not freeze_text_encoder:
        text_lora_parameters_one_with_lr = {'params':
            text_lora_parameters_one, 'weight_decay': args.
            adam_weight_decay_text_encoder if args.
            adam_weight_decay_text_encoder else args.adam_weight_decay,
            'lr': args.text_encoder_lr if args.text_encoder_lr else args.
            learning_rate}
        text_lora_parameters_two_with_lr = {'params':
            text_lora_parameters_two, 'weight_decay': args.
            adam_weight_decay_text_encoder if args.
            adam_weight_decay_text_encoder else args.adam_weight_decay,
            'lr': args.text_encoder_lr if args.text_encoder_lr else args.
            learning_rate}
        params_to_optimize = [unet_lora_parameters_with_lr,
            text_lora_parameters_one_with_lr, text_lora_parameters_two_with_lr]
    else:
        params_to_optimize = [unet_lora_parameters_with_lr]
    if not (args.optimizer.lower() == 'prodigy' or args.optimizer.lower() ==
        'adamw'):
        logger.warning(
            f'Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy].Defaulting to adamW'
            )
        args.optimizer = 'adamw'
    if args.use_8bit_adam and not args.optimizer.lower() == 'adamw':
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was set to {args.optimizer.lower()}"
            )
    if args.optimizer.lower() == 'adamw':
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
        optimizer = optimizer_class(params_to_optimize, betas=(args.
            adam_beta1, args.adam_beta2), weight_decay=args.
            adam_weight_decay, eps=args.adam_epsilon)
    if args.optimizer.lower() == 'prodigy':
        try:
            import prodigyopt
        except ImportError:
            raise ImportError(
                'To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`'
                )
        optimizer_class = prodigyopt.Prodigy
        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
                )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warning(
                f'Learning rates were provided both for the unet and the text encoder- e.g. text_encoder_lr: {args.text_encoder_lr} and learning_rate: {args.learning_rate}. When using prodigy only learning_rate is used as the initial learning rate.'
                )
            params_to_optimize[1]['lr'] = args.learning_rate
            params_to_optimize[2]['lr'] = args.learning_rate
        optimizer = optimizer_class(params_to_optimize, lr=args.
            learning_rate, betas=(args.adam_beta1, args.adam_beta2), beta3=
            args.prodigy_beta3, weight_decay=args.adam_weight_decay, eps=
            args.adam_epsilon, decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup)
    train_dataset = DreamBoothDataset(instance_data_root=args.
        instance_data_dir, instance_prompt=args.instance_prompt,
        class_prompt=args.class_prompt, dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name, cache_dir=args.
        cache_dir, image_column=args.image_column, train_text_encoder_ti=
        args.train_text_encoder_ti, caption_column=args.caption_column,
        class_data_root=args.class_data_dir if args.with_prior_preservation
         else None, token_abstraction_dict=token_abstraction_dict if args.
        train_text_encoder_ti else None, class_num=args.num_class_images,
        size=args.resolution, repeats=args.repeats, center_crop=args.
        center_crop)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.train_batch_size, shuffle=True, collate_fn=lambda
        examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers)

    def compute_time_ids(crops_coords_top_left, original_size=None):
        if original_size is None:
            original_size = args.resolution, args.resolution
        target_size = args.resolution, args.resolution
        add_time_ids = list(original_size + crops_coords_top_left + target_size
            )
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids
    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders, tokenizers, prompt)
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.
                    device)
            return prompt_embeds, pooled_prompt_embeds
    if freeze_text_encoder and not train_dataset.custom_instance_prompts:
        instance_prompt_hidden_states, instance_pooled_prompt_embeds = (
            compute_text_embeddings(args.instance_prompt, text_encoders,
            tokenizers))
    if args.with_prior_preservation:
        if freeze_text_encoder:
            class_prompt_hidden_states, class_pooled_prompt_embeds = (
                compute_text_embeddings(args.class_prompt, text_encoders,
                tokenizers))
    if freeze_text_encoder and not train_dataset.custom_instance_prompts:
        del tokenizers, text_encoders
        gc.collect()
        torch.cuda.empty_cache()
    add_special_tokens = True if args.train_text_encoder_ti else False
    if not train_dataset.custom_instance_prompts:
        if freeze_text_encoder:
            prompt_embeds = instance_prompt_hidden_states
            unet_add_text_embeds = instance_pooled_prompt_embeds
            if args.with_prior_preservation:
                prompt_embeds = torch.cat([prompt_embeds,
                    class_prompt_hidden_states], dim=0)
                unet_add_text_embeds = torch.cat([unet_add_text_embeds,
                    class_pooled_prompt_embeds], dim=0)
        else:
            tokens_one = tokenize_prompt(tokenizer_one, args.
                instance_prompt, add_special_tokens)
            tokens_two = tokenize_prompt(tokenizer_two, args.
                instance_prompt, add_special_tokens)
            if args.with_prior_preservation:
                class_tokens_one = tokenize_prompt(tokenizer_one, args.
                    class_prompt, add_special_tokens)
                class_tokens_two = tokenize_prompt(tokenizer_two, args.
                    class_prompt, add_special_tokens)
                tokens_one = torch.cat([tokens_one, class_tokens_one], dim=0)
                tokens_two = torch.cat([tokens_two, class_tokens_two], dim=0)
    if args.cache_latents:
        latents_cache = []
        for batch in tqdm(train_dataloader, desc='Caching latents'):
            with torch.no_grad():
                batch['pixel_values'] = batch['pixel_values'].to(accelerator
                    .device, non_blocking=True, dtype=torch.float32)
                latents_cache.append(vae.encode(batch['pixel_values']).
                    latent_dist)
        if args.validation_prompt is None:
            del vae
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
    if not freeze_text_encoder:
        (unet, text_encoder_one, text_encoder_two, optimizer,
            train_dataloader, lr_scheduler) = (accelerator.prepare(unet,
            text_encoder_one, text_encoder_two, optimizer, train_dataloader,
            lr_scheduler))
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
    args.num_train_epochs = math.ceil(args.max_train_steps /
        num_update_steps_per_epoch)
    if accelerator.is_main_process:
        tracker_name = ('dreambooth-lora-sd-xl' if 'playground' not in args
            .pretrained_model_name_or_path else 'dreambooth-lora-playground')
        accelerator.init_trackers(tracker_name, config=vars(args))
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

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype
            =dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in
            timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    if args.train_text_encoder:
        num_train_epochs_text_encoder = int(args.train_text_encoder_frac *
            args.num_train_epochs)
    elif args.train_text_encoder_ti:
        num_train_epochs_text_encoder = int(args.train_text_encoder_ti_frac *
            args.num_train_epochs)
    pivoted = False
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder or args.train_text_encoder_ti:
            if epoch == num_train_epochs_text_encoder:
                print('PIVOT HALFWAY', epoch)
                pivoted = True
            else:
                text_encoder_one.train()
                text_encoder_two.train()
                if args.train_text_encoder:
                    text_encoder_one.text_model.embeddings.requires_grad_(True)
                    text_encoder_two.text_model.embeddings.requires_grad_(True)
        for step, batch in enumerate(train_dataloader):
            if pivoted:
                optimizer.param_groups[1]['lr'] = 0.0
                optimizer.param_groups[2]['lr'] = 0.0
            with accelerator.accumulate(unet):
                prompts = batch['prompts']
                if train_dataset.custom_instance_prompts:
                    if freeze_text_encoder:
                        prompt_embeds, unet_add_text_embeds = (
                            compute_text_embeddings(prompts, text_encoders,
                            tokenizers))
                    else:
                        tokens_one = tokenize_prompt(tokenizer_one, prompts,
                            add_special_tokens)
                        tokens_two = tokenize_prompt(tokenizer_two, prompts,
                            add_special_tokens)
                if args.cache_latents:
                    model_input = latents_cache[step].sample()
                else:
                    pixel_values = batch['pixel_values'].to(dtype=vae.dtype)
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                if latents_mean is None and latents_std is None:
                    model_input = model_input * vae.config.scaling_factor
                    if args.pretrained_vae_model_name_or_path is None:
                        model_input = model_input.to(weight_dtype)
                else:
                    latents_mean = latents_mean.to(device=model_input.
                        device, dtype=model_input.dtype)
                    latents_std = latents_std.to(device=model_input.device,
                        dtype=model_input.dtype)
                    model_input = (model_input - latents_mean
                        ) * vae.config.scaling_factor / latents_std
                    model_input = model_input.to(dtype=weight_dtype)
                noise = torch.randn_like(model_input)
                if args.noise_offset:
                    noise += args.noise_offset * torch.randn((model_input.
                        shape[0], model_input.shape[1], 1, 1), device=
                        model_input.device)
                bsz = model_input.shape[0]
                if not args.do_edm_style_training:
                    timesteps = torch.randint(0, noise_scheduler.config.
                        num_train_timesteps, (bsz,), device=model_input.device)
                    timesteps = timesteps.long()
                else:
                    indices = torch.randint(0, noise_scheduler.config.
                        num_train_timesteps, (bsz,))
                    timesteps = noise_scheduler.timesteps[indices].to(device
                        =model_input.device)
                noisy_model_input = noise_scheduler.add_noise(model_input,
                    noise, timesteps)
                if args.do_edm_style_training:
                    sigmas = get_sigmas(timesteps, len(noisy_model_input.
                        shape), noisy_model_input.dtype)
                    if 'EDM' in scheduler_type:
                        inp_noisy_latents = (noise_scheduler.
                            precondition_inputs(noisy_model_input, sigmas))
                    else:
                        inp_noisy_latents = noisy_model_input / (sigmas ** 
                            2 + 1) ** 0.5
                add_time_ids = torch.cat([compute_time_ids(original_size=s,
                    crops_coords_top_left=c) for s, c in zip(batch[
                    'original_sizes'], batch['crop_top_lefts'])])
                if not train_dataset.custom_instance_prompts:
                    elems_to_repeat_text_embeds = (bsz // 2 if args.
                        with_prior_preservation else bsz)
                else:
                    elems_to_repeat_text_embeds = 1
                if freeze_text_encoder:
                    unet_added_conditions = {'time_ids': add_time_ids,
                        'text_embeds': unet_add_text_embeds.repeat(
                        elems_to_repeat_text_embeds, 1)}
                    prompt_embeds_input = prompt_embeds.repeat(
                        elems_to_repeat_text_embeds, 1, 1)
                    model_pred = unet(inp_noisy_latents if args.
                        do_edm_style_training else noisy_model_input,
                        timesteps, prompt_embeds_input, added_cond_kwargs=
                        unet_added_conditions, return_dict=False)[0]
                else:
                    unet_added_conditions = {'time_ids': add_time_ids}
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=None, prompt=None, text_input_ids_list=[
                        tokens_one, tokens_two])
                    unet_added_conditions.update({'text_embeds':
                        pooled_prompt_embeds.repeat(
                        elems_to_repeat_text_embeds, 1)})
                    prompt_embeds_input = prompt_embeds.repeat(
                        elems_to_repeat_text_embeds, 1, 1)
                    model_pred = unet(inp_noisy_latents if args.
                        do_edm_style_training else noisy_model_input,
                        timesteps, prompt_embeds_input, added_cond_kwargs=
                        unet_added_conditions, return_dict=False)[0]
                weighting = None
                if args.do_edm_style_training:
                    if 'EDM' in scheduler_type:
                        model_pred = noise_scheduler.precondition_outputs(
                            noisy_model_input, model_pred, sigmas)
                    elif noise_scheduler.config.prediction_type == 'epsilon':
                        model_pred = model_pred * -sigmas + noisy_model_input
                    elif noise_scheduler.config.prediction_type == 'v_prediction':
                        model_pred = model_pred * (-sigmas / (sigmas ** 2 +
                            1) ** 0.5) + noisy_model_input / (sigmas ** 2 + 1)
                    if 'EDM' not in scheduler_type:
                        weighting = (sigmas ** -2.0).float()
                if noise_scheduler.config.prediction_type == 'epsilon':
                    target = (model_input if args.do_edm_style_training else
                        noise)
                elif noise_scheduler.config.prediction_type == 'v_prediction':
                    target = (model_input if args.do_edm_style_training else
                        noise_scheduler.get_velocity(model_input, noise,
                        timesteps))
                else:
                    raise ValueError(
                        f'Unknown prediction type {noise_scheduler.config.prediction_type}'
                        )
                if args.with_prior_preservation:
                    model_pred, model_pred_prior = torch.chunk(model_pred, 
                        2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    if weighting is not None:
                        prior_loss = torch.mean((weighting.float() * (
                            model_pred_prior.float() - target_prior.float()
                            ) ** 2).reshape(target_prior.shape[0], -1), 1)
                        prior_loss = prior_loss.mean()
                    else:
                        prior_loss = F.mse_loss(model_pred_prior.float(),
                            target_prior.float(), reduction='mean')
                if args.snr_gamma is None:
                    if weighting is not None:
                        loss = torch.mean((weighting.float() * (model_pred.
                            float() - target.float()) ** 2).reshape(target.
                            shape[0], -1), 1)
                        loss = loss.mean()
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(
                            ), reduction='mean')
                else:
                    if args.with_prior_preservation:
                        snr_timesteps, _ = torch.chunk(timesteps, 2, dim=0)
                    else:
                        snr_timesteps = timesteps
                    snr = compute_snr(noise_scheduler, snr_timesteps)
                    base_weight = torch.stack([snr, args.snr_gamma * torch.
                        ones_like(snr_timesteps)], dim=1).min(dim=1)[0] / snr
                    if (noise_scheduler.config.prediction_type ==
                        'v_prediction'):
                        mse_loss_weights = base_weight + 1
                    else:
                        mse_loss_weights = base_weight
                    loss = F.mse_loss(model_pred.float(), target.float(),
                        reduction='none')
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))
                        ) * mse_loss_weights
                    loss = loss.mean()
                if args.with_prior_preservation:
                    loss = loss + args.prior_loss_weight * prior_loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (itertools.chain(unet_lora_parameters,
                        text_lora_parameters_one, text_lora_parameters_two) if
                        args.train_text_encoder or args.
                        train_text_encoder_ti else unet_lora_parameters)
                    accelerator.clip_grad_norm_(params_to_clip, args.
                        max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if args.train_text_encoder_ti:
                    embedding_handler.retract_embeddings()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
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
            logs = {'loss': loss.detach().item(), 'lr': lr_scheduler.
                get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
        if accelerator.is_main_process:
            if (args.validation_prompt is not None and epoch % args.
                validation_epochs == 0):
                logger.info(
                    f"""Running validation... 
 Generating {args.num_validation_images} images with prompt: {args.validation_prompt}."""
                    )
                if freeze_text_encoder:
                    text_encoder_one = text_encoder_cls_one.from_pretrained(
                        args.pretrained_model_name_or_path, subfolder=
                        'text_encoder', revision=args.revision, variant=
                        args.variant)
                    text_encoder_two = text_encoder_cls_two.from_pretrained(
                        args.pretrained_model_name_or_path, subfolder=
                        'text_encoder_2', revision=args.revision, variant=
                        args.variant)
                pipeline = StableDiffusionXLPipeline.from_pretrained(args.
                    pretrained_model_name_or_path, vae=vae, tokenizer=
                    tokenizer_one, tokenizer_2=tokenizer_two, text_encoder=
                    accelerator.unwrap_model(text_encoder_one),
                    text_encoder_2=accelerator.unwrap_model(
                    text_encoder_two), unet=accelerator.unwrap_model(unet),
                    revision=args.revision, variant=args.variant,
                    torch_dtype=weight_dtype)
                scheduler_args = {}
                if not args.do_edm_style_training:
                    if 'variance_type' in pipeline.scheduler.config:
                        variance_type = pipeline.scheduler.config.variance_type
                        if variance_type in ['learned', 'learned_range']:
                            variance_type = 'fixed_small'
                        scheduler_args['variance_type'] = variance_type
                    pipeline.scheduler = (DPMSolverMultistepScheduler.
                        from_config(pipeline.scheduler.config, **
                        scheduler_args))
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)
                generator = torch.Generator(device=accelerator.device
                    ).manual_seed(args.seed) if args.seed else None
                pipeline_args = {'prompt': args.validation_prompt}
                if torch.backends.mps.is_available(
                    ) or 'playground' in args.pretrained_model_name_or_path:
                    autocast_ctx = nullcontext()
                else:
                    autocast_ctx = torch.autocast(accelerator.device.type)
                with autocast_ctx:
                    images = [pipeline(**pipeline_args, generator=generator
                        ).images[0] for _ in range(args.num_validation_images)]
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
        unet = accelerator.unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_layers = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unet))
        if args.train_text_encoder:
            text_encoder_one = accelerator.unwrap_model(text_encoder_one)
            text_encoder_lora_layers = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(text_encoder_one.to(torch.float32)))
            text_encoder_two = accelerator.unwrap_model(text_encoder_two)
            text_encoder_2_lora_layers = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(text_encoder_two.to(torch.float32)))
        else:
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None
        StableDiffusionXLPipeline.save_lora_weights(save_directory=args.
            output_dir, unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers)
        if args.train_text_encoder_ti:
            embeddings_path = (
                f'{args.output_dir}/{args.output_dir}_emb.safetensors')
            embedding_handler.save_embeddings(embeddings_path)
        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            vae = AutoencoderKL.from_pretrained(vae_path, subfolder='vae' if
                args.pretrained_vae_model_name_or_path is None else None,
                revision=args.revision, variant=args.variant, torch_dtype=
                weight_dtype)
            pipeline = StableDiffusionXLPipeline.from_pretrained(args.
                pretrained_model_name_or_path, vae=vae, revision=args.
                revision, variant=args.variant, torch_dtype=weight_dtype)
            scheduler_args = {}
            if not args.do_edm_style_training:
                if 'variance_type' in pipeline.scheduler.config:
                    variance_type = pipeline.scheduler.config.variance_type
                    if variance_type in ['learned', 'learned_range']:
                        variance_type = 'fixed_small'
                    scheduler_args['variance_type'] = variance_type
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipeline.scheduler.config, **scheduler_args)
            pipeline.load_lora_weights(args.output_dir)
            if args.train_text_encoder_ti:
                state_dict = load_file(embeddings_path)
                all_new_tokens = []
                for key, value in token_abstraction_dict.items():
                    all_new_tokens.extend(value)
                pipeline.load_textual_inversion(state_dict['clip_l'], token
                    =all_new_tokens, text_encoder=pipeline.text_encoder,
                    tokenizer=pipeline.tokenizer)
                pipeline.load_textual_inversion(state_dict['clip_g'], token
                    =all_new_tokens, text_encoder=pipeline.text_encoder_2,
                    tokenizer=pipeline.tokenizer_2)
            pipeline = pipeline.to(accelerator.device)
            generator = torch.Generator(device=accelerator.device).manual_seed(
                args.seed) if args.seed else None
            images = [pipeline(args.validation_prompt, num_inference_steps=
                25, generator=generator).images[0] for _ in range(args.
                num_validation_images)]
            for tracker in accelerator.trackers:
                if tracker.name == 'tensorboard':
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images('test', np_images, epoch,
                        dataformats='NHWC')
                if tracker.name == 'wandb':
                    tracker.log({'test': [wandb.Image(image, caption=
                        f'{i}: {args.validation_prompt}') for i, image in
                        enumerate(images)]})
        lora_state_dict = load_file(
            f'{args.output_dir}/pytorch_lora_weights.safetensors')
        peft_state_dict = convert_all_state_dict_to_peft(lora_state_dict)
        kohya_state_dict = convert_state_dict_to_kohya(peft_state_dict)
        save_file(kohya_state_dict,
            f'{args.output_dir}/{args.output_dir}.safetensors')
        save_model_card(model_id if not args.push_to_hub else repo_id,
            use_dora=args.use_dora, images=images, base_model=args.
            pretrained_model_name_or_path, train_text_encoder=args.
            train_text_encoder, train_text_encoder_ti=args.
            train_text_encoder_ti, token_abstraction_dict=train_dataset.
            token_abstraction_dict, instance_prompt=args.instance_prompt,
            validation_prompt=args.validation_prompt, repo_folder=args.
            output_dir, vae_path=args.pretrained_vae_model_name_or_path)
        if args.push_to_hub:
            upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message='End of training', ignore_patterns=['step_*',
                'epoch_*'])
    accelerator.end_training()
