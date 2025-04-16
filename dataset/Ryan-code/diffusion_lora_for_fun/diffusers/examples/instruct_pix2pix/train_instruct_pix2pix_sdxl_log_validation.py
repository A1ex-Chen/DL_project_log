def log_validation(pipeline, args, accelerator, generator, global_step,
    is_final_validation=False):
    logger.info(
        f"""Running validation... 
 Generating {args.num_validation_images} images with prompt: {args.validation_prompt}."""
        )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    val_save_dir = os.path.join(args.output_dir, 'validation_images')
    if not os.path.exists(val_save_dir):
        os.makedirs(val_save_dir)
    original_image = (lambda image_url_or_path: load_image(
        image_url_or_path) if urlparse(image_url_or_path).scheme else Image
        .open(image_url_or_path).convert('RGB'))(args.val_image_url_or_path)
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)
    with autocast_ctx:
        edited_images = []
        for val_img_idx in range(args.num_validation_images):
            a_val_img = pipeline(args.validation_prompt, image=
                original_image, num_inference_steps=20,
                image_guidance_scale=1.5, guidance_scale=7, generator=generator
                ).images[0]
            edited_images.append(a_val_img)
            a_val_img.save(os.path.join(val_save_dir,
                f'step_{global_step}_val_img_{val_img_idx}.png'))
    for tracker in accelerator.trackers:
        if tracker.name == 'wandb':
            wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
            for edited_image in edited_images:
                wandb_table.add_data(wandb.Image(original_image), wandb.
                    Image(edited_image), args.validation_prompt)
            logger_name = 'test' if is_final_validation else 'validation'
            tracker.log({logger_name: wandb_table})
