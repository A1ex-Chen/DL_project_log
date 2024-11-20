def main():
    args = parse_args()
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
        pretrained_model_name_or_path, subfolder='scheduler')
    tokenizer = CLIPTokenizer.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='tokenizer', revision=args
        .revision)
    text_encoder = CLIPTextModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='text_encoder', revision=
        args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
        subfolder='vae', revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='unet', revision=args.revision
        )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    if args.use_peft:
        from peft import LoraConfig, LoraModel, get_peft_model_state_dict, set_peft_model_state_dict
        UNET_TARGET_MODULES = ['to_q', 'to_v', 'query', 'value']
        TEXT_ENCODER_TARGET_MODULES = ['q_proj', 'v_proj']
        config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=UNET_TARGET_MODULES, lora_dropout=args.
            lora_dropout, bias=args.lora_bias)
        unet = LoraModel(config, unet)
        vae.requires_grad_(False)
        if args.train_text_encoder:
            config = LoraConfig(r=args.lora_text_encoder_r, lora_alpha=args
                .lora_text_encoder_alpha, target_modules=
                TEXT_ENCODER_TARGET_MODULES, lora_dropout=args.
                lora_text_encoder_dropout, bias=args.lora_text_encoder_bias)
            text_encoder = LoraModel(config, text_encoder)
    else:
        unet.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
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
            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=
                hidden_size, cross_attention_dim=cross_attention_dim)
        unet.set_attn_processor(lora_attn_procs)
        lora_layers = AttnProcsLayers(unet.attn_processors)
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
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
    if args.use_peft:
        params_to_optimize = itertools.chain(unet.parameters(),
            text_encoder.parameters()
            ) if args.train_text_encoder else unet.parameters()
        optimizer = optimizer_cls(params_to_optimize, lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.
            adam_weight_decay, eps=args.adam_epsilon)
    else:
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
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0
            ] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
                )
    if args.caption_column is None:
        caption_column = dataset_columns[1
            ] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
                )

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else
                    caption[0])
            else:
                raise ValueError(
                    f'Caption column `{caption_column}` should contain either strings or lists of strings.'
                    )
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length,
            padding='max_length', truncation=True, return_tensors='pt')
        return inputs.input_ids
    train_transforms = transforms.Compose([transforms.Resize(args.
        resolution, interpolation=transforms.InterpolationMode.BILINEAR), 
        transforms.CenterCrop(args.resolution) if args.center_crop else
        transforms.RandomCrop(args.resolution), transforms.
        RandomHorizontalFlip() if args.random_flip else transforms.Lambda(
        lambda x: x), transforms.ToTensor(), transforms.Normalize([0.5], [
        0.5])])

    def preprocess_train(examples):
        images = [image.convert('RGB') for image in examples[image_column]]
        examples['pixel_values'] = [train_transforms(image) for image in images
            ]
        examples['input_ids'] = tokenize_captions(examples)
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
        input_ids = torch.stack([example['input_ids'] for example in examples])
        return {'pixel_values': pixel_values, 'input_ids': input_ids}
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
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes)
    if args.use_peft:
        if args.train_text_encoder:
            (unet, text_encoder, optimizer, train_dataloader, lr_scheduler) = (
                accelerator.prepare(unet, text_encoder, optimizer,
                train_dataloader, lr_scheduler))
        else:
            unet, optimizer, train_dataloader, lr_scheduler = (accelerator.
                prepare(unet, optimizer, train_dataloader, lr_scheduler))
    else:
        lora_layers, optimizer, train_dataloader, lr_scheduler = (accelerator
            .prepare(lora_layers, optimizer, train_dataloader, lr_scheduler))
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
        if args.train_text_encoder:
            text_encoder.train()
        train_loss = 0.0
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
                timesteps = torch.randint(0, noise_scheduler.config.
                    num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise,
                    timesteps)
                encoder_hidden_states = text_encoder(batch['input_ids'])[0]
                if noise_scheduler.config.prediction_type == 'epsilon':
                    target = noise
                elif noise_scheduler.config.prediction_type == 'v_prediction':
                    target = noise_scheduler.get_velocity(latents, noise,
                        timesteps)
                else:
                    raise ValueError(
                        f'Unknown prediction type {noise_scheduler.config.prediction_type}'
                        )
                model_pred = unet(noisy_latents, timesteps,
                    encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(),
                    reduction='mean')
                avg_loss = accelerator.gather(loss.repeat(args.
                    train_batch_size)).mean()
                train_loss += avg_loss.item(
                    ) / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.use_peft:
                        params_to_clip = itertools.chain(unet.parameters(),
                            text_encoder.parameters()
                            ) if args.train_text_encoder else unet.parameters()
                    else:
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
                pipeline = DiffusionPipeline.from_pretrained(args.
                    pretrained_model_name_or_path, unet=accelerator.
                    unwrap_model(unet), text_encoder=accelerator.
                    unwrap_model(text_encoder), revision=args.revision,
                    torch_dtype=weight_dtype)
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)
                generator = torch.Generator(device=accelerator.device
                    ).manual_seed(args.seed)
                images = []
                for _ in range(args.num_validation_images):
                    images.append(pipeline(args.validation_prompt,
                        num_inference_steps=30, generator=generator).images[0])
                if accelerator.is_main_process:
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
        if args.use_peft:
            lora_config = {}
            unwarpped_unet = accelerator.unwrap_model(unet)
            state_dict = get_peft_model_state_dict(unwarpped_unet,
                state_dict=accelerator.get_state_dict(unet))
            lora_config['peft_config'
                ] = unwarpped_unet.get_peft_config_as_dict(inference=True)
            if args.train_text_encoder:
                unwarpped_text_encoder = accelerator.unwrap_model(text_encoder)
                text_encoder_state_dict = get_peft_model_state_dict(
                    unwarpped_text_encoder, state_dict=accelerator.
                    get_state_dict(text_encoder))
                text_encoder_state_dict = {f'text_encoder_{k}': v for k, v in
                    text_encoder_state_dict.items()}
                state_dict.update(text_encoder_state_dict)
                lora_config['text_encoder_peft_config'
                    ] = unwarpped_text_encoder.get_peft_config_as_dict(
                    inference=True)
            accelerator.save(state_dict, os.path.join(args.output_dir,
                f'{global_step}_lora.pt'))
            with open(os.path.join(args.output_dir,
                f'{global_step}_lora_config.json'), 'w') as f:
                json.dump(lora_config, f)
        else:
            unet = unet.to(torch.float32)
            unet.save_attn_procs(args.output_dir)
        if args.push_to_hub:
            save_model_card(repo_id, images=images, base_model=args.
                pretrained_model_name_or_path, dataset_name=args.
                dataset_name, repo_folder=args.output_dir)
            upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message='End of training', ignore_patterns=['step_*',
                'epoch_*'])
    pipeline = DiffusionPipeline.from_pretrained(args.
        pretrained_model_name_or_path, revision=args.revision, torch_dtype=
        weight_dtype)
    if args.use_peft:

        def load_and_set_lora_ckpt(pipe, ckpt_dir, global_step, device, dtype):
            with open(os.path.join(args.output_dir,
                f'{global_step}_lora_config.json'), 'r') as f:
                lora_config = json.load(f)
            print(lora_config)
            checkpoint = os.path.join(args.output_dir, f'{global_step}_lora.pt'
                )
            lora_checkpoint_sd = torch.load(checkpoint)
            unet_lora_ds = {k: v for k, v in lora_checkpoint_sd.items() if 
                'text_encoder_' not in k}
            text_encoder_lora_ds = {k.replace('text_encoder_', ''): v for k,
                v in lora_checkpoint_sd.items() if 'text_encoder_' in k}
            unet_config = LoraConfig(**lora_config['peft_config'])
            pipe.unet = LoraModel(unet_config, pipe.unet)
            set_peft_model_state_dict(pipe.unet, unet_lora_ds)
            if 'text_encoder_peft_config' in lora_config:
                text_encoder_config = LoraConfig(**lora_config[
                    'text_encoder_peft_config'])
                pipe.text_encoder = LoraModel(text_encoder_config, pipe.
                    text_encoder)
                set_peft_model_state_dict(pipe.text_encoder,
                    text_encoder_lora_ds)
            if dtype in (torch.float16, torch.bfloat16):
                pipe.unet.half()
                pipe.text_encoder.half()
            pipe.to(device)
            return pipe
        pipeline = load_and_set_lora_ckpt(pipeline, args.output_dir,
            global_step, accelerator.device, weight_dtype)
    else:
        pipeline = pipeline.to(accelerator.device)
        pipeline.unet.load_attn_procs(args.output_dir)
    if args.seed is not None:
        generator = torch.Generator(device=accelerator.device).manual_seed(args
            .seed)
    else:
        generator = None
    images = []
    for _ in range(args.num_validation_images):
        images.append(pipeline(args.validation_prompt, num_inference_steps=
            30, generator=generator).images[0])
    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if tracker.name == 'tensorboard':
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images('test', np_images, epoch,
                    dataformats='NHWC')
            if tracker.name == 'wandb':
                tracker.log({'test': [wandb.Image(image, caption=
                    f'{i}: {args.validation_prompt}') for i, image in
                    enumerate(images)]})
    accelerator.end_training()
