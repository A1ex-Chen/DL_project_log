def log_validation(vae, image_encoder, image_processor, unet, args,
    accelerator, weight_dtype, epoch):
    logger.info('Running validation... ')
    pipeline = AutoPipelineForText2Image.from_pretrained(args.
        pretrained_decoder_model_name_or_path, vae=accelerator.unwrap_model
        (vae), prior_image_encoder=accelerator.unwrap_model(image_encoder),
        prior_image_processor=image_processor, unet=accelerator.
        unwrap_model(unet), torch_dtype=weight_dtype)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args
            .seed)
    images = []
    for i in range(len(args.validation_prompts)):
        with torch.autocast('cuda'):
            image = pipeline(args.validation_prompts[i],
                num_inference_steps=20, generator=generator).images[0]
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
