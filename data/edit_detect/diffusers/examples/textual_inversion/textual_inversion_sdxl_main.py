def main():
    args = parse_args()
    if args.report_to == 'wandb' and args.hub_token is not None:
        raise ValueError(
            'You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token. Please use `huggingface-cli login` to authenticate with the Hub.'
            )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
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
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.
                output_dir).name, exist_ok=True, token=args.hub_token).repo_id
    tokenizer_1 = CLIPTokenizer.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='tokenizer')
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='tokenizer_2')
    noise_scheduler = DDPMScheduler.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='scheduler')
    text_encoder_1 = CLIPTextModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='text_encoder', revision=
        args.revision)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='text_encoder_2', revision
        =args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
        subfolder='vae', revision=args.revision, variant=args.variant)
    unet = UNet2DConditionModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='unet', revision=args.
        revision, variant=args.variant)
    placeholder_tokens = [args.placeholder_token]
    if args.num_vectors < 1:
        raise ValueError(
            f'--num_vectors has to be larger or equal to 1, but is {args.num_vectors}'
            )
    additional_tokens = []
    for i in range(1, args.num_vectors):
        additional_tokens.append(f'{args.placeholder_token}_{i}')
    placeholder_tokens += additional_tokens
    num_added_tokens = tokenizer_1.add_tokens(placeholder_tokens)
    if num_added_tokens != args.num_vectors:
        raise ValueError(
            f'The tokenizer already contains the token {args.placeholder_token}. Please pass a different `placeholder_token` that is not already in the tokenizer.'
            )
    token_ids = tokenizer_1.encode(args.initializer_token,
        add_special_tokens=False)
    if len(token_ids) > 1:
        raise ValueError('The initializer token must be a single token.')
    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer_1.convert_tokens_to_ids(
        placeholder_tokens)
    text_encoder_1.resize_token_embeddings(len(tokenizer_1))
    token_embeds = text_encoder_1.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    text_encoder_1.text_model.encoder.requires_grad_(False)
    text_encoder_1.text_model.final_layer_norm.requires_grad_(False)
    text_encoder_1.text_model.embeddings.position_embedding.requires_grad_(
        False)
    if args.gradient_checkpointing:
        text_encoder_1.gradient_checkpointing_enable()
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
                'To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.'
                )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(text_encoder_1.get_input_embeddings().
        parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.
        adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon
        )
    placeholder_token = ' '.join(tokenizer_1.convert_ids_to_tokens(
        placeholder_token_ids))
    train_dataset = TextualInversionDataset(data_root=args.train_data_dir,
        tokenizer_1=tokenizer_1, tokenizer_2=tokenizer_2, size=args.
        resolution, placeholder_token=placeholder_token, repeats=args.
        repeats, learnable_property=args.learnable_property, center_crop=
        args.center_crop, set='train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.train_batch_size, shuffle=True, num_workers=args.
        dataloader_num_workers)
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
        num_cycles=args.lr_num_cycles)
    text_encoder_1.train()
    text_encoder_1, optimizer, train_dataloader, lr_scheduler = (accelerator
        .prepare(text_encoder_1, optimizer, train_dataloader, lr_scheduler))
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
    args.num_train_epochs = math.ceil(args.max_train_steps /
        num_update_steps_per_epoch)
    if accelerator.is_main_process:
        accelerator.init_trackers('textual_inversion', config=vars(args))
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
    orig_embeds_params = accelerator.unwrap_model(text_encoder_1
        ).get_input_embeddings().weight.data.clone()
    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder_1.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder_1):
                latents = vae.encode(batch['pixel_values'].to(dtype=
                    weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.
                    num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise,
                    timesteps)
                encoder_hidden_states_1 = text_encoder_1(batch[
                    'input_ids_1'], output_hidden_states=True).hidden_states[-2
                    ].to(dtype=weight_dtype)
                encoder_output_2 = text_encoder_2(batch['input_ids_2'].
                    reshape(batch['input_ids_1'].shape[0], -1),
                    output_hidden_states=True)
                encoder_hidden_states_2 = encoder_output_2.hidden_states[-2
                    ].to(dtype=weight_dtype)
                original_size = [(batch['original_size'][0][i].item(),
                    batch['original_size'][1][i].item()) for i in range(
                    args.train_batch_size)]
                crop_top_left = [(batch['crop_top_left'][0][i].item(),
                    batch['crop_top_left'][1][i].item()) for i in range(
                    args.train_batch_size)]
                target_size = args.resolution, args.resolution
                add_time_ids = torch.cat([torch.tensor(original_size[i] +
                    crop_top_left[i] + target_size) for i in range(args.
                    train_batch_size)]).to(accelerator.device, dtype=
                    weight_dtype)
                added_cond_kwargs = {'text_embeds': encoder_output_2[0],
                    'time_ids': add_time_ids}
                encoder_hidden_states = torch.cat([encoder_hidden_states_1,
                    encoder_hidden_states_2], dim=-1)
                model_pred = unet(noisy_latents, timesteps,
                    encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
                    ).sample
                if noise_scheduler.config.prediction_type == 'epsilon':
                    target = noise
                elif noise_scheduler.config.prediction_type == 'v_prediction':
                    target = noise_scheduler.get_velocity(latents, noise,
                        timesteps)
                else:
                    raise ValueError(
                        f'Unknown prediction type {noise_scheduler.config.prediction_type}'
                        )
                loss = F.mse_loss(model_pred.float(), target.float(),
                    reduction='mean')
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                index_no_updates = torch.ones((len(tokenizer_1),), dtype=
                    torch.bool)
                index_no_updates[min(placeholder_token_ids):max(
                    placeholder_token_ids) + 1] = False
                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder_1
                        ).get_input_embeddings().weight[index_no_updates
                        ] = orig_embeds_params[index_no_updates]
            if accelerator.sync_gradients:
                images = []
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    weight_name = (
                        f'learned_embeds-steps-{global_step}.safetensors')
                    save_path = os.path.join(args.output_dir, weight_name)
                    save_progress(text_encoder_1, placeholder_token_ids,
                        accelerator, args, save_path, safe_serialization=True)
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
                    if (args.validation_prompt is not None and global_step %
                        args.validation_steps == 0):
                        images = log_validation(text_encoder_1,
                            text_encoder_2, tokenizer_1, tokenizer_2, unet,
                            vae, args, accelerator, weight_dtype, epoch)
            logs = {'loss': loss.detach().item(), 'lr': lr_scheduler.
                get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.validation_prompt:
            images = log_validation(text_encoder_1, text_encoder_2,
                tokenizer_1, tokenizer_2, unet, vae, args, accelerator,
                weight_dtype, epoch, is_final_validation=True)
        if args.push_to_hub and not args.save_as_full_pipeline:
            logger.warning(
                'Enabling full model saving because --push_to_hub=True was specified.'
                )
            save_full_model = True
        else:
            save_full_model = args.save_as_full_pipeline
        if save_full_model:
            pipeline = DiffusionPipeline.from_pretrained(args.
                pretrained_model_name_or_path, text_encoder=accelerator.
                unwrap_model(text_encoder_1), text_encoder_2=text_encoder_2,
                vae=vae, unet=unet, tokenizer=tokenizer_1, tokenizer_2=
                tokenizer_2)
            pipeline.save_pretrained(args.output_dir)
        weight_name = 'learned_embeds.safetensors'
        save_path = os.path.join(args.output_dir, weight_name)
        save_progress(text_encoder_1, placeholder_token_ids, accelerator,
            args, save_path, safe_serialization=True)
        if args.push_to_hub:
            save_model_card(repo_id, images=images, base_model=args.
                pretrained_model_name_or_path, repo_folder=args.output_dir)
            upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message='End of training', ignore_patterns=['step_*',
                'epoch_*'])
    accelerator.end_training()
