def get_optimizer_and_scheduler(args, model, train_dataloader, accelerator):
    if args.use_lora:
        assert not args.finetune_vae, 'Fine-tuning VAE with LoRA has not been implemented.'
        lora_layers = diffusers.loaders.AttnProcsLayers(model.student_unet.
            attn_processors)
        optimizer_parameters = lora_layers.parameters()
        logger.info('Optimizing LoRA parameters.')
    elif args.finetune_vae:
        optimizer_parameters = list(model.student_unet.parameters()) + list(
            model.vae.decoder.parameters()) + list(model.vae.
            post_quant_conv.parameters())
        logger.info('Optimizing UNet and VAE decoder parameters.')
    else:
        optimizer_parameters = model.student_unet.parameters()
        logger.info('Optimizing UNet parameters.')
    num_unet_trainable_parameters = sum(p.numel() for p in model.
        student_unet.parameters() if p.requires_grad)
    num_vae_trainable_parameters = sum(p.numel() for p in model.vae.
        parameters() if p.requires_grad)
    num_total_trainable_parameters = sum(p.numel() for p in model.
        parameters() if p.requires_grad)
    num_other_trainable_parameters = (num_total_trainable_parameters -
        num_unet_trainable_parameters - num_vae_trainable_parameters)
    logger.info(
        f'Num trainable U-Net parameters: {num_unet_trainable_parameters}.')
    logger.info(
        f'Num trainable VAE parameters: {num_vae_trainable_parameters}.')
    logger.info(
        f'Num trainable other parameters: {num_other_trainable_parameters}.')
    logger.info(
        f'Num trainable total parameters: {num_total_trainable_parameters}.')
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.
        learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    optimizer.zero_grad()
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.
        gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = (args.num_train_epochs *
            num_update_steps_per_epoch)
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=
        optimizer, num_warmup_steps=args.num_warmup_steps * accelerator.
        num_processes, num_training_steps=args.max_train_steps)
    return optimizer, lr_scheduler, overrode_max_train_steps
