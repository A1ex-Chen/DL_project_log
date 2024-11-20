def log_validation(args, unet, vae, accelerator, weight_dtype, epoch,
    is_final_validation=False):
    logger.info(
        f'Running validation... \n Generating images with prompts:\n {VALIDATION_PROMPTS}.'
        )
    if is_final_validation:
        if args.mixed_precision == 'fp16':
            vae.to(weight_dtype)
    pipeline = DiffusionPipeline.from_pretrained(args.
        pretrained_model_name_or_path, vae=vae, revision=args.revision,
        variant=args.variant, torch_dtype=weight_dtype)
    if not is_final_validation:
        pipeline.unet = accelerator.unwrap_model(unet)
    else:
        pipeline.load_lora_weights(args.output_dir, weight_name=
            'pytorch_lora_weights.safetensors')
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device).manual_seed(args
        .seed) if args.seed else None
    images = []
    context = contextlib.nullcontext(
        ) if is_final_validation else torch.cuda.amp.autocast()
    guidance_scale = 5.0
    num_inference_steps = 25
    for prompt in VALIDATION_PROMPTS:
        with context:
            image = pipeline(prompt, num_inference_steps=
                num_inference_steps, guidance_scale=guidance_scale,
                generator=generator).images[0]
            images.append(image)
    tracker_key = 'test' if is_final_validation else 'validation'
    for tracker in accelerator.trackers:
        if tracker.name == 'tensorboard':
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(tracker_key, np_images, epoch,
                dataformats='NHWC')
        if tracker.name == 'wandb':
            tracker.log({tracker_key: [wandb.Image(image, caption=
                f'{i}: {VALIDATION_PROMPTS[i]}') for i, image in enumerate(
                images)]})
    if is_final_validation:
        pipeline.disable_lora()
        generator = torch.Generator(device=accelerator.device).manual_seed(args
            .seed) if args.seed else None
        no_lora_images = [pipeline(prompt, num_inference_steps=
            num_inference_steps, guidance_scale=guidance_scale, generator=
            generator).images[0] for prompt in VALIDATION_PROMPTS]
        for tracker in accelerator.trackers:
            if tracker.name == 'tensorboard':
                np_images = np.stack([np.asarray(img) for img in
                    no_lora_images])
                tracker.writer.add_images('test_without_lora', np_images,
                    epoch, dataformats='NHWC')
            if tracker.name == 'wandb':
                tracker.log({'test_without_lora': [wandb.Image(image,
                    caption=f'{i}: {VALIDATION_PROMPTS[i]}') for i, image in
                    enumerate(no_lora_images)]})
