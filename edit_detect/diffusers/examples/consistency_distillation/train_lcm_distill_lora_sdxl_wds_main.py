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
        log_with=args.report_to, project_config=accelerator_project_config,
        split_batches=True)
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
                output_dir).name, exist_ok=True, token=args.hub_token,
                private=True).repo_id
    noise_scheduler = DDPMScheduler.from_pretrained(args.
        pretrained_teacher_model, subfolder='scheduler', revision=args.
        teacher_revision)
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    solver = DDIMSolver(noise_scheduler.alphas_cumprod.numpy(), timesteps=
        noise_scheduler.config.num_train_timesteps, ddim_timesteps=args.
        num_ddim_timesteps)
    tokenizer_one = AutoTokenizer.from_pretrained(args.
        pretrained_teacher_model, subfolder='tokenizer', revision=args.
        teacher_revision, use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(args.
        pretrained_teacher_model, subfolder='tokenizer_2', revision=args.
        teacher_revision, use_fast=False)
    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.
        pretrained_teacher_model, args.teacher_revision)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(args.
        pretrained_teacher_model, args.teacher_revision, subfolder=
        'text_encoder_2')
    text_encoder_one = text_encoder_cls_one.from_pretrained(args.
        pretrained_teacher_model, subfolder='text_encoder', revision=args.
        teacher_revision)
    text_encoder_two = text_encoder_cls_two.from_pretrained(args.
        pretrained_teacher_model, subfolder='text_encoder_2', revision=args
        .teacher_revision)
    vae_path = (args.pretrained_teacher_model if args.
        pretrained_vae_model_name_or_path is None else args.
        pretrained_vae_model_name_or_path)
    vae = AutoencoderKL.from_pretrained(vae_path, subfolder='vae' if args.
        pretrained_vae_model_name_or_path is None else None, revision=args.
        teacher_revision)
    teacher_unet = UNet2DConditionModel.from_pretrained(args.
        pretrained_teacher_model, subfolder='unet', revision=args.
        teacher_revision)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    teacher_unet.requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(args.
        pretrained_teacher_model, subfolder='unet', revision=args.
        teacher_revision)
    unet.train()
    low_precision_error_string = (
        ' Please make sure to always have all model weights in full float32 precision when starting training - even if doing mixed precision training, copy of the weights should still be float32.'
        )
    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f'Controlnet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}'
            )
    if args.lora_target_modules is not None:
        lora_target_modules = [module_key.strip() for module_key in args.
            lora_target_modules.split(',')]
    else:
        lora_target_modules = ['to_q', 'to_k', 'to_v', 'to_out.0',
            'proj_in', 'proj_out', 'ff.net.0.proj', 'ff.net.2', 'conv1',
            'conv2', 'conv_shortcut', 'downsamplers.0.conv',
            'upsamplers.0.conv', 'time_emb_proj']
    lora_config = LoraConfig(r=args.lora_rank, target_modules=
        lora_target_modules, lora_alpha=args.lora_alpha, lora_dropout=args.
        lora_dropout)
    unet = get_peft_model(unet, lora_config)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device)
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    teacher_unet.to(accelerator.device)
    if args.cast_teacher_unet:
        teacher_unet.to(dtype=weight_dtype)
    alpha_schedule = alpha_schedule.to(accelerator.device)
    sigma_schedule = sigma_schedule.to(accelerator.device)
    solver = solver.to(accelerator.device)
    if version.parse(accelerate.__version__) >= version.parse('0.16.0'):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                unet_ = accelerator.unwrap_model(unet)
                lora_state_dict = get_peft_model_state_dict(unet_,
                    adapter_name='default')
                StableDiffusionXLPipeline.save_lora_weights(os.path.join(
                    output_dir, 'unet_lora'), lora_state_dict)
                unet_.save_pretrained(output_dir)
                for _, model in enumerate(models):
                    weights.pop()

        def load_model_hook(models, input_dir):
            unet_ = accelerator.unwrap_model(unet)
            unet_.load_adapter(input_dir, 'default', is_trainable=True)
            for _ in range(len(models)):
                models.pop()
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
        else:
            raise ValueError(
                'xformers is not available. Make sure it is installed correctly'
                )
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
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
    optimizer = optimizer_class(unet.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.
        adam_weight_decay, eps=args.adam_epsilon)

    def compute_embeddings(prompt_batch, original_sizes, crop_coords,
        proportion_empty_prompts, text_encoders, tokenizers, is_train=True):
        target_size = args.resolution, args.resolution
        original_sizes = list(map(list, zip(*original_sizes)))
        crops_coords_top_left = list(map(list, zip(*crop_coords)))
        original_sizes = torch.tensor(original_sizes, dtype=torch.long)
        crops_coords_top_left = torch.tensor(crops_coords_top_left, dtype=
            torch.long)
        prompt_embeds, pooled_prompt_embeds = encode_prompt(prompt_batch,
            text_encoders, tokenizers, proportion_empty_prompts, is_train)
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = list(target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
        add_time_ids = torch.cat([original_sizes, crops_coords_top_left,
            add_time_ids], dim=-1)
        add_time_ids = add_time_ids.to(accelerator.device, dtype=
            prompt_embeds.dtype)
        prompt_embeds = prompt_embeds.to(accelerator.device)
        add_text_embeds = add_text_embeds.to(accelerator.device)
        unet_added_cond_kwargs = {'text_embeds': add_text_embeds,
            'time_ids': add_time_ids}
        return {'prompt_embeds': prompt_embeds, **unet_added_cond_kwargs}
    dataset = SDXLText2ImageDataset(train_shards_path_or_url=args.
        train_shards_path_or_url, num_train_examples=args.max_train_samples,
        per_gpu_batch_size=args.train_batch_size, global_batch_size=args.
        train_batch_size * accelerator.num_processes, num_workers=args.
        dataloader_num_workers, resolution=args.resolution,
        interpolation_type=args.interpolation_type, shuffle_buffer_size=
        1000, pin_memory=True, persistent_workers=True,
        use_fix_crop_and_size=args.use_fix_crop_and_size)
    train_dataloader = dataset.train_dataloader
    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]
    compute_embeddings_fn = functools.partial(compute_embeddings,
        proportion_empty_prompts=0, text_encoders=text_encoders, tokenizers
        =tokenizers)
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches /
        args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
        overrode_max_train_steps = True
    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.
            gradient_accumulation_steps * args.train_batch_size *
            accelerator.num_processes)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps, num_training_steps=args.
        max_train_steps)
    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer,
        lr_scheduler)
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches /
        args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
    args.num_train_epochs = math.ceil(args.max_train_steps /
        num_update_steps_per_epoch)
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=
            tracker_config)
    uncond_prompt_embeds = torch.zeros(args.train_batch_size, 77, 2048).to(
        accelerator.device)
    uncond_pooled_prompt_embeds = torch.zeros(args.train_batch_size, 1280).to(
        accelerator.device)
    total_batch_size = (args.train_batch_size * accelerator.num_processes *
        args.gradient_accumulation_steps)
    logger.info('***** Running training *****')
    logger.info(f'  Num batches each epoch = {train_dataloader.num_batches}')
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
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                image, text, orig_size, crop_coords = batch
                image = image.to(accelerator.device, non_blocking=True)
                encoded_text = compute_embeddings_fn(text, orig_size,
                    crop_coords)
                if args.pretrained_vae_model_name_or_path is not None:
                    pixel_values = image.to(dtype=weight_dtype)
                    if vae.dtype != weight_dtype:
                        vae.to(dtype=weight_dtype)
                else:
                    pixel_values = image
                latents = []
                for i in range(0, pixel_values.shape[0], args.
                    vae_encode_batch_size):
                    latents.append(vae.encode(pixel_values[i:i + args.
                        vae_encode_batch_size]).latent_dist.sample())
                latents = torch.cat(latents, dim=0)
                latents = latents * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    latents = latents.to(weight_dtype)
                bsz = latents.shape[0]
                topk = (noise_scheduler.config.num_train_timesteps // args.
                    num_ddim_timesteps)
                index = torch.randint(0, args.num_ddim_timesteps, (bsz,),
                    device=latents.device).long()
                start_timesteps = solver.ddim_timesteps[index]
                timesteps = start_timesteps - topk
                timesteps = torch.where(timesteps < 0, torch.zeros_like(
                    timesteps), timesteps)
                c_skip_start, c_out_start = scalings_for_boundary_conditions(
                    start_timesteps, timestep_scaling=args.
                    timestep_scaling_factor)
                c_skip_start, c_out_start = [append_dims(x, latents.ndim) for
                    x in [c_skip_start, c_out_start]]
                c_skip, c_out = scalings_for_boundary_conditions(timesteps,
                    timestep_scaling=args.timestep_scaling_factor)
                c_skip, c_out = [append_dims(x, latents.ndim) for x in [
                    c_skip, c_out]]
                noise = torch.randn_like(latents)
                noisy_model_input = noise_scheduler.add_noise(latents,
                    noise, start_timesteps)
                w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min
                w = w.reshape(bsz, 1, 1, 1)
                w = w.to(device=latents.device, dtype=latents.dtype)
                prompt_embeds = encoded_text.pop('prompt_embeds')
                noise_pred = unet(noisy_model_input, start_timesteps,
                    timestep_cond=None, encoder_hidden_states=prompt_embeds
                    .float(), added_cond_kwargs=encoded_text).sample
                pred_x_0 = get_predicted_original_sample(noise_pred,
                    start_timesteps, noisy_model_input, noise_scheduler.
                    config.prediction_type, alpha_schedule, sigma_schedule)
                model_pred = (c_skip_start * noisy_model_input + 
                    c_out_start * pred_x_0)
                with torch.no_grad():
                    if torch.backends.mps.is_available(
                        ) or 'playground' in args.pretrained_model_name_or_path:
                        autocast_ctx = nullcontext()
                    else:
                        autocast_ctx = torch.autocast(accelerator.device.type)
                    with autocast_ctx:
                        cond_teacher_output = teacher_unet(noisy_model_input
                            .to(weight_dtype), start_timesteps,
                            encoder_hidden_states=prompt_embeds.to(
                            weight_dtype), added_cond_kwargs={k: v.to(
                            weight_dtype) for k, v in encoded_text.items()}
                            ).sample
                        cond_pred_x0 = get_predicted_original_sample(
                            cond_teacher_output, start_timesteps,
                            noisy_model_input, noise_scheduler.config.
                            prediction_type, alpha_schedule, sigma_schedule)
                        cond_pred_noise = get_predicted_noise(
                            cond_teacher_output, start_timesteps,
                            noisy_model_input, noise_scheduler.config.
                            prediction_type, alpha_schedule, sigma_schedule)
                        uncond_added_conditions = copy.deepcopy(encoded_text)
                        uncond_added_conditions['text_embeds'
                            ] = uncond_pooled_prompt_embeds
                        uncond_teacher_output = teacher_unet(noisy_model_input
                            .to(weight_dtype), start_timesteps,
                            encoder_hidden_states=uncond_prompt_embeds.to(
                            weight_dtype), added_cond_kwargs={k: v.to(
                            weight_dtype) for k, v in
                            uncond_added_conditions.items()}).sample
                        uncond_pred_x0 = get_predicted_original_sample(
                            uncond_teacher_output, start_timesteps,
                            noisy_model_input, noise_scheduler.config.
                            prediction_type, alpha_schedule, sigma_schedule)
                        uncond_pred_noise = get_predicted_noise(
                            uncond_teacher_output, start_timesteps,
                            noisy_model_input, noise_scheduler.config.
                            prediction_type, alpha_schedule, sigma_schedule)
                        pred_x0 = cond_pred_x0 + w * (cond_pred_x0 -
                            uncond_pred_x0)
                        pred_noise = cond_pred_noise + w * (cond_pred_noise -
                            uncond_pred_noise)
                        x_prev = solver.ddim_step(pred_x0, pred_noise, index)
                with torch.no_grad():
                    if torch.backends.mps.is_available():
                        autocast_ctx = nullcontext()
                    else:
                        autocast_ctx = torch.autocast(accelerator.device.
                            type, dtype=weight_dtype)
                    with autocast_ctx:
                        target_noise_pred = unet(x_prev.float(), timesteps,
                            timestep_cond=None, encoder_hidden_states=
                            prompt_embeds.float(), added_cond_kwargs=
                            encoded_text).sample
                    pred_x_0 = get_predicted_original_sample(target_noise_pred,
                        timesteps, x_prev, noise_scheduler.config.
                        prediction_type, alpha_schedule, sigma_schedule)
                    target = c_skip * x_prev + c_out * pred_x_0
                if args.loss_type == 'l2':
                    loss = F.mse_loss(model_pred.float(), target.float(),
                        reduction='mean')
                elif args.loss_type == 'huber':
                    loss = torch.mean(torch.sqrt((model_pred.float() -
                        target.float()) ** 2 + args.huber_c ** 2) - args.
                        huber_c)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.
                        max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
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
                    if global_step % args.validation_steps == 0:
                        log_validation(vae, unet, args, accelerator,
                            weight_dtype, global_step)
            logs = {'loss': loss.detach().item(), 'lr': lr_scheduler.
                get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(args.output_dir)
        lora_state_dict = get_peft_model_state_dict(unet, adapter_name=
            'default')
        StableDiffusionXLPipeline.save_lora_weights(os.path.join(args.
            output_dir, 'unet_lora'), lora_state_dict)
        if args.push_to_hub:
            upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message='End of training', ignore_patterns=['step_*',
                'epoch_*'])
    accelerator.end_training()
