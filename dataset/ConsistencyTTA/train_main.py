def main():
    args = parse_args()
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs['log_with'] = args.report_to
        accelerator_log_kwargs['logging_dir'] = args.output_dir
    accelerator = Accelerator(gradient_accumulation_steps=args.
        gradient_accumulation_steps, mixed_precision='bf16' if args.
        use_bf16 else 'no', **accelerator_log_kwargs)
    logging.basicConfig(format=
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt=
        '%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    datasets.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        seed_all(args.seed)
        set_seed(args.seed)
    if accelerator.is_main_process:
        if args.output_dir is None or args.output_dir == '':
            args.output_dir = f'saved/{int(time())}'
            if not os.path.exists('saved'):
                os.makedirs('saved')
            os.makedirs(args.output_dir, exist_ok=True)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f'{args.output_dir}/outputs', exist_ok=True)
        with open(f'{args.output_dir}/summary.jsonl', 'a') as f:
            f.write(json.dumps(dict(vars(args))) + '\n\n')
        accelerator.project_configuration.automatic_checkpoint_naming = False
        wandb.init(project=f'Text to Audio Stage-{args.stage} Distillation')
    accelerator.wait_for_everyone()
    pretrained_model_name = 'audioldm-s-full'
    vae, stft = build_pretrained_models(pretrained_model_name)
    vae.eval()
    vae.requires_grad_(False)
    stft.eval()
    stft.requires_grad_(False)
    if args.stage == 1:
        model_class = AudioGDM
    elif args.finetune_vae:
        model_class = AudioLCM_FTVAE
    else:
        model_class = AudioLCM
    model = model_class(text_encoder_name=args.text_encoder_name,
        scheduler_name=args.scheduler_name, unet_model_name=args.
        unet_model_name, unet_model_config_path=args.unet_model_config,
        snr_gamma=args.snr_gamma, freeze_text_encoder=args.
        freeze_text_encoder, uncondition=args.uncondition, use_edm=args.
        use_edm, use_karras=args.use_karras, use_lora=args.use_lora,
        target_ema_decay=args.target_ema_decay, ema_decay=args.ema_decay,
        num_diffusion_steps=args.num_diffusion_steps,
        teacher_guidance_scale=args.teacher_guidance_scale, loss_type=args.
        loss_type, vae=vae)
    if args.tango_model is not None:
        model.load_state_dict_from_tango(tango_state_dict=torch.load(args.
            tango_model, map_location='cpu'), stage1_state_dict=torch.load(
            args.stage1_model, map_location='cpu') if args.stage1_model is not
            None else None)
        logger.info(f'Loaded TANGO checkpoint from: {args.tango_model}')
        if args.stage == 2 and args.stage1_model is not None:
            logger.info(f'Loaded stage-1 checkpoint from: {args.stage1_model}')
    else:
        raise NotImplementedError
    assert args.freeze_text_encoder, 'Text encoder funetuning has not been implemented.'
    assert args.unet_model_config, 'unet_model_config must be specified.'
    model.text_encoder.eval()
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    logger.info('Text encoder is frozen.')
    train_dataloader, eval_dataloader, test_dataloader = get_dataloaders(args,
        accelerator)
    optimizer, lr_scheduler, overrode_max_train_steps = (
        get_optimizer_and_scheduler(args, model, train_dataloader, accelerator)
        )
    vae, stft, model = vae.cuda(), stft.cuda(), model.cuda()
    vae, stft, model, optimizer, lr_scheduler = accelerator.prepare(vae,
        stft, model, optimizer, lr_scheduler)
    train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader, test_dataloader)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
    args.num_train_epochs = math.ceil(args.max_train_steps /
        num_update_steps_per_epoch)
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)
    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config['lr_scheduler_type'] = experiment_config[
            'lr_scheduler_type'].value
        accelerator.init_trackers('text_to_audio_diffusion', experiment_config)
    total_batch_size = (args.per_device_train_batch_size * accelerator.
        num_processes * args.gradient_accumulation_steps)
    logger.info(f'***** Running stage-{args.stage} training *****')
    logger.info(f'  Num examples = {len(train_dataloader.dataset)}')
    logger.info(f'  Num Epochs = {args.num_train_epochs}')
    logger.info(
        f'  Instantaneous batch size per device = {args.per_device_train_batch_size}'
        )
    logger.info(
        f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}'
        )
    logger.info(
        f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}')
    logger.info(f'  Total optimization steps = {args.max_train_steps}')
    if args.resume_from_checkpoint:
        if (args.resume_from_checkpoint is not None or args.
            resume_from_checkpoint != ''):
            accelerator.load_state(args.resume_from_checkpoint,
                map_location='cpu')
            logger.info(
                f'Resumed from local checkpoint: {args.resume_from_checkpoint}'
                )
        else:
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
    progress_bar = tqdm(range(args.max_train_steps), disable=not
        accelerator.is_local_main_process)
    starting_epoch = args.starting_epoch
    completed_steps = starting_epoch * num_update_steps_per_epoch
    progress_bar.update(completed_steps - progress_bar.n)
    best_eval_loss = np.inf
    if args.eval_first:
        total_val_loss = eval_model(model=model, vae=vae, stft=stft, stage=
            args.stage, eval_dataloader=eval_dataloader, accelerator=
            accelerator, num_diffusion_steps=args.num_diffusion_steps)
        save_checkpoint, best_eval_loss = log_results(accelerator=
            accelerator, logger=logger, epoch=0, completed_steps=
            completed_steps, lr=optimizer.param_groups[0]['lr'], train_loss
            =None, val_loss=total_val_loss, best_eval_loss=best_eval_loss,
            output_dir=args.output_dir, with_tracking=args.with_tracking)
    for epoch in range(starting_epoch, args.num_train_epochs):
        total_loss, completed_steps = train_one_epoch(model=model, vae=vae,
            stft=stft, train_dataloader=train_dataloader, accelerator=
            accelerator, args=args, optimizer=optimizer, lr_scheduler=
            lr_scheduler, checkpointing_steps=checkpointing_steps,
            completed_steps=completed_steps, progress_bar=progress_bar)
        total_val_loss = eval_model(model=model, vae=vae, stft=stft, stage=
            args.stage, eval_dataloader=eval_dataloader, accelerator=
            accelerator, num_diffusion_steps=args.num_diffusion_steps)
        accelerator.wait_for_everyone()
        save_checkpoint, best_eval_loss = log_results(accelerator=
            accelerator, logger=logger, epoch=epoch + 1, completed_steps=
            completed_steps, lr=optimizer.param_groups[0]['lr'], train_loss
            =total_loss / len(train_dataloader), val_loss=total_val_loss,
            best_eval_loss=best_eval_loss, output_dir=args.output_dir,
            with_tracking=args.with_tracking)
        model_saved = False
        while not model_saved:
            try:
                if (accelerator.is_main_process and args.
                    checkpointing_steps == 'best'):
                    if save_checkpoint:
                        accelerator.save_state(f'{args.output_dir}/best')
                    if (epoch + 1) % args.save_every == 0:
                        accelerator.save_state(
                            f'{args.output_dir}/epoch_{epoch + 1}')
                if (accelerator.is_main_process and args.
                    checkpointing_steps == 'epoch'):
                    accelerator.save_state(
                        f'{args.output_dir}/epoch_{epoch + 1}')
                model_saved = True
                logger.info('Model successfully saved.')
            except:
                logger.info('Save model failed. Retrying.')
