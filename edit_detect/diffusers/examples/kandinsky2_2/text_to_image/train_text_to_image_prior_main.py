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
    noise_scheduler = DDPMScheduler(beta_schedule='squaredcos_cap_v2',
        prediction_type='sample')
    image_processor = CLIPImageProcessor.from_pretrained(args.
        pretrained_prior_model_name_or_path, subfolder='image_processor')
    tokenizer = CLIPTokenizer.from_pretrained(args.
        pretrained_prior_model_name_or_path, subfolder='tokenizer')

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState(
            ).deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []
        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.
            pretrained_prior_model_name_or_path, subfolder='image_encoder',
            torch_dtype=weight_dtype).eval()
        text_encoder = CLIPTextModelWithProjection.from_pretrained(args.
            pretrained_prior_model_name_or_path, subfolder='text_encoder',
            torch_dtype=weight_dtype).eval()
    prior = PriorTransformer.from_pretrained(args.
        pretrained_prior_model_name_or_path, subfolder='prior')
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    prior.train()
    if args.use_ema:
        ema_prior = PriorTransformer.from_pretrained(args.
            pretrained_prior_model_name_or_path, subfolder='prior')
        ema_prior = EMAModel(ema_prior.parameters(), model_cls=
            PriorTransformer, model_config=ema_prior.config)
        ema_prior.to(accelerator.device)
    if version.parse(accelerate.__version__) >= version.parse('0.16.0'):

        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_prior.save_pretrained(os.path.join(output_dir, 'prior_ema')
                    )
            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, 'prior'))
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(
                    input_dir, 'prior_ema'), PriorTransformer)
                ema_prior.load_state_dict(load_model.state_dict())
                ema_prior.to(accelerator.device)
                del load_model
            for i in range(len(models)):
                model = models.pop()
                load_model = PriorTransformer.from_pretrained(input_dir,
                    subfolder='prior')
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
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
    optimizer = optimizer_cls(prior.parameters(), lr=args.learning_rate,
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
        text_input_ids = inputs.input_ids
        text_mask = inputs.attention_mask.bool()
        return text_input_ids, text_mask

    def preprocess_train(examples):
        images = [image.convert('RGB') for image in examples[image_column]]
        examples['clip_pixel_values'] = image_processor(images,
            return_tensors='pt').pixel_values
        examples['text_input_ids'], examples['text_mask'] = tokenize_captions(
            examples)
        return examples
    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset['train'] = dataset['train'].shuffle(seed=args.seed).select(
                range(args.max_train_samples))
        train_dataset = dataset['train'].with_transform(preprocess_train)

    def collate_fn(examples):
        clip_pixel_values = torch.stack([example['clip_pixel_values'] for
            example in examples])
        clip_pixel_values = clip_pixel_values.to(memory_format=torch.
            contiguous_format).float()
        text_input_ids = torch.stack([example['text_input_ids'] for example in
            examples])
        text_mask = torch.stack([example['text_mask'] for example in examples])
        return {'clip_pixel_values': clip_pixel_values, 'text_input_ids':
            text_input_ids, 'text_mask': text_mask}
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
    clip_mean = prior.clip_mean.clone()
    clip_std = prior.clip_std.clone()
    prior, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        prior, optimizer, train_dataloader, lr_scheduler)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
    args.num_train_epochs = math.ceil(args.max_train_steps /
        num_update_steps_per_epoch)
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop('validation_prompts')
        accelerator.init_trackers(args.tracker_project_name, tracker_config)
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
    clip_mean = clip_mean.to(weight_dtype).to(accelerator.device)
    clip_std = clip_std.to(weight_dtype).to(accelerator.device)
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(prior):
                text_input_ids, text_mask, clip_images = batch['text_input_ids'
                    ], batch['text_mask'], batch['clip_pixel_values'].to(
                    weight_dtype)
                with torch.no_grad():
                    text_encoder_output = text_encoder(text_input_ids)
                    prompt_embeds = text_encoder_output.text_embeds
                    text_encoder_hidden_states = (text_encoder_output.
                        last_hidden_state)
                    image_embeds = image_encoder(clip_images).image_embeds
                    noise = torch.randn_like(image_embeds)
                    bsz = image_embeds.shape[0]
                    timesteps = torch.randint(0, noise_scheduler.config.
                        num_train_timesteps, (bsz,), device=image_embeds.device
                        )
                    timesteps = timesteps.long()
                    image_embeds = (image_embeds - clip_mean) / clip_std
                    noisy_latents = noise_scheduler.add_noise(image_embeds,
                        noise, timesteps)
                    target = image_embeds
                model_pred = prior(noisy_latents, timestep=timesteps,
                    proj_embedding=prompt_embeds, encoder_hidden_states=
                    text_encoder_hidden_states, attention_mask=text_mask
                    ).predicted_image_embedding
                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(),
                        reduction='mean')
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma *
                        torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == 'epsilon':
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == 'v_prediction':
                        mse_loss_weights = mse_loss_weights / (snr + 1)
                    loss = F.mse_loss(model_pred.float(), target.float(),
                        reduction='none')
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))
                        ) * mse_loss_weights
                    loss = loss.mean()
                avg_loss = accelerator.gather(loss.repeat(args.
                    train_batch_size)).mean()
                train_loss += avg_loss.item(
                    ) / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(prior.parameters(), args.
                        max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_prior.step(prior.parameters())
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
            if global_step >= args.max_train_steps:
                break
        if accelerator.is_main_process:
            if (args.validation_prompts is not None and epoch % args.
                validation_epochs == 0):
                if args.use_ema:
                    ema_prior.store(prior.parameters())
                    ema_prior.copy_to(prior.parameters())
                log_validation(image_encoder, image_processor, text_encoder,
                    tokenizer, prior, args, accelerator, weight_dtype,
                    global_step)
                if args.use_ema:
                    ema_prior.restore(prior.parameters())
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        prior = accelerator.unwrap_model(prior)
        if args.use_ema:
            ema_prior.copy_to(prior.parameters())
        pipeline = AutoPipelineForText2Image.from_pretrained(args.
            pretrained_decoder_model_name_or_path, prior_image_encoder=
            image_encoder, prior_text_encoder=text_encoder, prior_prior=prior)
        pipeline.prior_pipe.save_pretrained(args.output_dir)
        images = []
        if args.validation_prompts is not None:
            logger.info('Running inference for collecting generated images...')
            pipeline = pipeline.to(accelerator.device)
            pipeline.torch_dtype = weight_dtype
            pipeline.set_progress_bar_config(disable=True)
            if args.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=accelerator.device
                    ).manual_seed(args.seed)
            for i in range(len(args.validation_prompts)):
                with torch.autocast('cuda'):
                    image = pipeline(args.validation_prompts[i],
                        num_inference_steps=20, generator=generator).images[0]
                images.append(image)
        if args.push_to_hub:
            save_model_card(args, repo_id, images, repo_folder=args.output_dir)
            upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message='End of training', ignore_patterns=['step_*',
                'epoch_*'])
    accelerator.end_training()
