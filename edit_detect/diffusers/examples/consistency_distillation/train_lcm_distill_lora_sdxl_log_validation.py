def log_validation(vae, args, accelerator, weight_dtype, step, unet=None,
    is_final_validation=False):
    logger.info('Running validation... ')
    pipeline = StableDiffusionXLPipeline.from_pretrained(args.
        pretrained_teacher_model, vae=vae, scheduler=LCMScheduler.
        from_pretrained(args.pretrained_teacher_model, subfolder=
        'scheduler'), revision=args.revision, torch_dtype=weight_dtype).to(
        accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    to_load = None
    if not is_final_validation:
        if unet is None:
            raise ValueError(
                'Must provide a `unet` when doing intermediate validation.')
        unet = accelerator.unwrap_model(unet)
        state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict
            (unet))
        to_load = state_dict
    else:
        to_load = args.output_dir
    pipeline.load_lora_weights(to_load)
    pipeline.fuse_lora()
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args
            .seed)
    validation_prompts = ['cute sundar pichai character',
        'robotic cat with wings', 'a photo of yoda',
        'a cute creature with blue eyes']
    image_logs = []
    for _, prompt in enumerate(validation_prompts):
        images = []
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type, dtype=
                weight_dtype)
        with autocast_ctx:
            images = pipeline(prompt=prompt, num_inference_steps=4,
                num_images_per_prompt=4, generator=generator,
                guidance_scale=0.0).images
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
            logger_name = 'test' if is_final_validation else 'validation'
            tracker.log({logger_name: formatted_images})
        else:
            logger.warning(f'image logging not implemented for {tracker.name}')
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        return image_logs
