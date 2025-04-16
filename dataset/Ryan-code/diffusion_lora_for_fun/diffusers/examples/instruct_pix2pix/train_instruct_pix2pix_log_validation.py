def log_validation(pipeline, args, accelerator, generator):
    logger.info(
        f"""Running validation... 
 Generating {args.num_validation_images} images with prompt: {args.validation_prompt}."""
        )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    original_image = download_image(args.val_image_url)
    edited_images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)
    with autocast_ctx:
        for _ in range(args.num_validation_images):
            edited_images.append(pipeline(args.validation_prompt, image=
                original_image, num_inference_steps=20,
                image_guidance_scale=1.5, guidance_scale=7, generator=
                generator).images[0])
    for tracker in accelerator.trackers:
        if tracker.name == 'wandb':
            wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
            for edited_image in edited_images:
                wandb_table.add_data(wandb.Image(original_image), wandb.
                    Image(edited_image), args.validation_prompt)
            tracker.log({'validation': wandb_table})
