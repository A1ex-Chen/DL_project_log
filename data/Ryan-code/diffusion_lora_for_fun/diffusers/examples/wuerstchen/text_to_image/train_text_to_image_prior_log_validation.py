def log_validation(text_encoder, tokenizer, prior, args, accelerator,
    weight_dtype, epoch):
    logger.info('Running validation... ')
    pipeline = AutoPipelineForText2Image.from_pretrained(args.
        pretrained_decoder_model_name_or_path, prior_prior=accelerator.
        unwrap_model(prior), prior_text_encoder=accelerator.unwrap_model(
        text_encoder), prior_tokenizer=tokenizer, torch_dtype=weight_dtype)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args
            .seed)
    images = []
    for i in range(len(args.validation_prompts)):
        with torch.autocast('cuda'):
            image = pipeline(args.validation_prompts[i], prior_timesteps=
                DEFAULT_STAGE_C_TIMESTEPS, generator=generator, height=args
                .resolution, width=args.resolution).images[0]
        images.append(image)
    for tracker in accelerator.trackers:
        if tracker.name == 'tensorboard':
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images('validation', np_images, epoch,
                dataformats='NHWC')
        elif tracker.name == 'wandb':
            tracker.log({'validation': [wandb.Image(image, caption=
                f'{i}: {args.validation_prompts[i]}') for i, image in
                enumerate(images)]})
        else:
            logger.warning(f'image logging not implemented for {tracker.name}')
    del pipeline
    torch.cuda.empty_cache()
    return images
