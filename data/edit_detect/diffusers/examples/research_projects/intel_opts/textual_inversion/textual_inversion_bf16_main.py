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
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.
                output_dir).name, exist_ok=True, token=args.hub_token).repo_id
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.
            pretrained_model_name_or_path, subfolder='tokenizer')
    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f'The tokenizer already contains the token {args.placeholder_token}. Please pass a different `placeholder_token` that is not already in the tokenizer.'
            )
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens
        =False)
    if len(token_ids) > 1:
        raise ValueError('The initializer token must be a single token.')
    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.
        placeholder_token)
    text_encoder = CLIPTextModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='text_encoder', revision=
        args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
        subfolder='vae', revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='unet', revision=args.revision
        )
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    params_to_freeze = itertools.chain(text_encoder.text_model.encoder.
        parameters(), text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters())
    freeze_params(params_to_freeze)
    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.
            gradient_accumulation_steps * args.train_batch_size *
            accelerator.num_processes)
    optimizer = torch.optim.AdamW(text_encoder.get_input_embeddings().
        parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.
        adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon
        )
    noise_scheduler = DDPMScheduler.from_pretrained(args.
        pretrained_model_name_or_path, subfolder='scheduler')
    train_dataset = TextualInversionDataset(data_root=args.train_data_dir,
        tokenizer=tokenizer, size=args.resolution, placeholder_token=args.
        placeholder_token, repeats=args.repeats, learnable_property=args.
        learnable_property, center_crop=args.center_crop, set='train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.train_batch_size, shuffle=True)
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
    text_encoder, optimizer, train_dataloader, lr_scheduler = (accelerator.
        prepare(text_encoder, optimizer, train_dataloader, lr_scheduler))
    vae.to(accelerator.device)
    unet.to(accelerator.device)
    vae.eval()
    unet.eval()
    unet = ipex.optimize(unet, dtype=torch.bfloat16, inplace=True)
    vae = ipex.optimize(vae, dtype=torch.bfloat16, inplace=True)
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
    progress_bar = tqdm(range(args.max_train_steps), disable=not
        accelerator.is_local_main_process)
    progress_bar.set_description('Steps')
    global_step = 0
    text_encoder.train()
    text_encoder, optimizer = ipex.optimize(text_encoder, optimizer=
        optimizer, dtype=torch.bfloat16)
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                with accelerator.accumulate(text_encoder):
                    latents = vae.encode(batch['pixel_values']
                        ).latent_dist.sample().detach()
                    latents = latents * vae.config.scaling_factor
                    noise = torch.randn(latents.shape).to(latents.device)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, noise_scheduler.config.
                        num_train_timesteps, (bsz,), device=latents.device
                        ).long()
                    noisy_latents = noise_scheduler.add_noise(latents,
                        noise, timesteps)
                    encoder_hidden_states = text_encoder(batch['input_ids'])[0]
                    model_pred = unet(noisy_latents, timesteps,
                        encoder_hidden_states).sample
                    if noise_scheduler.config.prediction_type == 'epsilon':
                        target = noise
                    elif noise_scheduler.config.prediction_type == 'v_prediction':
                        target = noise_scheduler.get_velocity(latents,
                            noise, timesteps)
                    else:
                        raise ValueError(
                            f'Unknown prediction type {noise_scheduler.config.prediction_type}'
                            )
                    loss = F.mse_loss(model_pred, target, reduction='none'
                        ).mean([1, 2, 3]).mean()
                    accelerator.backward(loss)
                    if accelerator.num_processes > 1:
                        grads = text_encoder.module.get_input_embeddings(
                            ).weight.grad
                    else:
                        grads = text_encoder.get_input_embeddings().weight.grad
                    index_grads_to_zero = torch.arange(len(tokenizer)
                        ) != placeholder_token_id
                    grads.data[index_grads_to_zero, :] = grads.data[
                        index_grads_to_zero, :].fill_(0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir,
                        f'learned_embeds-steps-{global_step}.bin')
                    save_progress(text_encoder, placeholder_token_id,
                        accelerator, args, save_path)
            logs = {'loss': loss.detach().item(), 'lr': lr_scheduler.
                get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.push_to_hub and args.only_save_embeds:
            logger.warning(
                'Enabling full model saving because --push_to_hub=True was specified.'
                )
            save_full_model = True
        else:
            save_full_model = not args.only_save_embeds
        if save_full_model:
            pipeline = StableDiffusionPipeline(text_encoder=accelerator.
                unwrap_model(text_encoder), vae=vae, unet=unet, tokenizer=
                tokenizer, scheduler=PNDMScheduler.from_pretrained(args.
                pretrained_model_name_or_path, subfolder='scheduler'),
                safety_checker=StableDiffusionSafetyChecker.from_pretrained
                ('CompVis/stable-diffusion-safety-checker'),
                feature_extractor=CLIPImageProcessor.from_pretrained(
                'openai/clip-vit-base-patch32'))
            pipeline.save_pretrained(args.output_dir)
        save_path = os.path.join(args.output_dir, 'learned_embeds.bin')
        save_progress(text_encoder, placeholder_token_id, accelerator, args,
            save_path)
        if args.push_to_hub:
            upload_folder(repo_id=repo_id, folder_path=args.output_dir,
                commit_message='End of training', ignore_patterns=['step_*',
                'epoch_*'])
    accelerator.end_training()
