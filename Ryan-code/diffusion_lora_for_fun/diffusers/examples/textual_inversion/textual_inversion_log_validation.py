def log_validation(text_encoder, tokenizer, unet, vae, args, accelerator,
    weight_dtype, epoch):
    logger.info(
        f"""Running validation... 
 Generating {args.num_validation_images} images with prompt: {args.validation_prompt}."""
        )
    pipeline = DiffusionPipeline.from_pretrained(args.
        pretrained_model_name_or_path, text_encoder=accelerator.
        unwrap_model(text_encoder), tokenizer=tokenizer, unet=unet, vae=vae,
        safety_checker=None, revision=args.revision, variant=args.variant,
        torch_dtype=weight_dtype)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.
        scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = None if args.seed is None else torch.Generator(device=
        accelerator.device).manual_seed(args.seed)
    images = []
    for _ in range(args.num_validation_images):
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type)
        with autocast_ctx:
            image = pipeline(args.validation_prompt, num_inference_steps=25,
                generator=generator).images[0]
        images.append(image)
    for tracker in accelerator.trackers:
        if tracker.name == 'tensorboard':
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images('validation', np_images, epoch,
                dataformats='NHWC')
        if tracker.name == 'wandb':
            tracker.log({'validation': [wandb.Image(image, caption=
                f'{i}: {args.validation_prompt}') for i, image in enumerate
                (images)]})
    del pipeline
    torch.cuda.empty_cache()
    return images
