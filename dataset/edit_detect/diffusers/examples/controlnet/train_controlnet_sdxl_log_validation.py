def log_validation(vae, unet, controlnet, args, accelerator, weight_dtype,
    step, is_final_validation=False):
    logger.info('Running validation... ')
    if not is_final_validation:
        controlnet = accelerator.unwrap_model(controlnet)
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(args
            .pretrained_model_name_or_path, vae=vae, unet=unet, controlnet=
            controlnet, revision=args.revision, variant=args.variant,
            torch_dtype=weight_dtype)
    else:
        controlnet = ControlNetModel.from_pretrained(args.output_dir,
            torch_dtype=weight_dtype)
        if args.pretrained_vae_model_name_or_path is not None:
            vae = AutoencoderKL.from_pretrained(args.
                pretrained_vae_model_name_or_path, torch_dtype=weight_dtype)
        else:
            vae = AutoencoderKL.from_pretrained(args.
                pretrained_model_name_or_path, subfolder='vae', torch_dtype
                =weight_dtype)
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(args
            .pretrained_model_name_or_path, vae=vae, controlnet=controlnet,
            revision=args.revision, variant=args.variant, torch_dtype=
            weight_dtype)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.
        scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args
            .seed)
    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image
            )
    else:
        raise ValueError(
            'number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`'
            )
    image_logs = []
    if is_final_validation or torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)
    for validation_prompt, validation_image in zip(validation_prompts,
        validation_images):
        validation_image = Image.open(validation_image).convert('RGB')
        validation_image = validation_image.resize((args.resolution, args.
            resolution))
        images = []
        for _ in range(args.num_validation_images):
            with autocast_ctx:
                image = pipeline(prompt=validation_prompt, image=
                    validation_image, num_inference_steps=20, generator=
                    generator).images[0]
            images.append(image)
        image_logs.append({'validation_image': validation_image, 'images':
            images, 'validation_prompt': validation_prompt})
    tracker_key = 'test' if is_final_validation else 'validation'
    for tracker in accelerator.trackers:
        if tracker.name == 'tensorboard':
            for log in image_logs:
                images = log['images']
                validation_prompt = log['validation_prompt']
                validation_image = log['validation_image']
                formatted_images = []
                formatted_images.append(np.asarray(validation_image))
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
                validation_image = log['validation_image']
                formatted_images.append(wandb.Image(validation_image,
                    caption='Controlnet conditioning'))
                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)
            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f'image logging not implemented for {tracker.name}')
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        return image_logs
