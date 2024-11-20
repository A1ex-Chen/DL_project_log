def log_validation(vae, unet, args, accelerator, weight_dtype, step, name=
    'target'):
    logger.info('Running validation... ')
    unet = accelerator.unwrap_model(unet)
    pipeline = StableDiffusionXLPipeline.from_pretrained(args.
        pretrained_teacher_model, vae=vae, unet=unet, scheduler=
        LCMScheduler.from_pretrained(args.pretrained_teacher_model,
        subfolder='scheduler'), revision=args.revision, torch_dtype=
        weight_dtype)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args
            .seed)
    validation_prompts = [
        'portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography'
        ,
        'Self-portrait oil painting, a beautiful cyborg with golden hair, 8k',
        'Astronaut in a jungle, cold color palette, muted colors, detailed, 8k'
        ,
        'A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece'
        ]
    image_logs = []
    for _, prompt in enumerate(validation_prompts):
        images = []
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type)
        with autocast_ctx:
            images = pipeline(prompt=prompt, num_inference_steps=4,
                num_images_per_prompt=4, generator=generator).images
        image_logs.append({'validation_prompt': prompt, 'images': images})
    for tracker in accelerator.trackers:
        if tracker.name == 'tensorboard':
            for log in image_logs:
                images = log['images']
                validation_prompt = log['validation_prompt']
                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image))
                formatted_images = np.stack(formatted_images)
                tracker.writer.add_images(validation_prompt,
                    formatted_images, step, dataformats='NHWC')
        elif tracker.name == 'wandb':
            formatted_images = []
            for log in image_logs:
                images = log['images']
                validation_prompt = log['validation_prompt']
                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)
            tracker.log({f'validation/{name}': formatted_images})
        else:
            logger.warning(f'image logging not implemented for {tracker.name}')
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        return image_logs
