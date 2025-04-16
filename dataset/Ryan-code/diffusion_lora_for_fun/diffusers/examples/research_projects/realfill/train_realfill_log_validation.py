def log_validation(text_encoder, tokenizer, unet, args, accelerator,
    weight_dtype, epoch):
    logger.info(
        f'Running validation... \nGenerating {args.num_validation_images} images'
        )
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(args.
        pretrained_model_name_or_path, tokenizer=tokenizer, revision=args.
        revision, torch_dtype=weight_dtype)
    pipeline.unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)
    pipeline.text_encoder = accelerator.unwrap_model(text_encoder,
        keep_fp32_wrapper=True)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.
        scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = None if args.seed is None else torch.Generator(device=
        accelerator.device).manual_seed(args.seed)
    target_dir = Path(args.train_data_dir) / 'target'
    target_image, target_mask = (target_dir / 'target.png', target_dir /
        'mask.png')
    image, mask_image = Image.open(target_image), Image.open(target_mask)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    images = []
    for _ in range(args.num_validation_images):
        image = pipeline(prompt='a photo of sks', image=image, mask_image=
            mask_image, num_inference_steps=25, guidance_scale=5, generator
            =generator).images[0]
        images.append(image)
    for tracker in accelerator.trackers:
        if tracker.name == 'tensorboard':
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images('validation', np_images, epoch,
                dataformats='NHWC')
        if tracker.name == 'wandb':
            tracker.log({'validation': [wandb.Image(image, caption=str(i)) for
                i, image in enumerate(images)]})
    del pipeline
    torch.cuda.empty_cache()
    return images
