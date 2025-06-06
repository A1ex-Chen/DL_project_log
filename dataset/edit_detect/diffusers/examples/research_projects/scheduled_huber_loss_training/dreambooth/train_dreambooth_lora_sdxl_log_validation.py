def log_validation(pipeline, args, accelerator, pipeline_args, epoch,
    is_final_validation=False):
    logger.info(
        f"""Running validation... 
 Generating {args.num_validation_images} images with prompt: {args.validation_prompt}."""
        )
    scheduler_args = {}
    if not args.do_edm_style_training:
        if 'variance_type' in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type
            if variance_type in ['learned', 'learned_range']:
                variance_type = 'fixed_small'
            scheduler_args['variance_type'] = variance_type
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline
            .scheduler.config, **scheduler_args)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device).manual_seed(args
        .seed) if args.seed else None
    inference_ctx = (contextlib.nullcontext() if 'playground' in args.
        pretrained_model_name_or_path else torch.cuda.amp.autocast())
    with inference_ctx:
        images = [pipeline(**pipeline_args, generator=generator).images[0] for
            _ in range(args.num_validation_images)]
    for tracker in accelerator.trackers:
        phase_name = 'test' if is_final_validation else 'validation'
        if tracker.name == 'tensorboard':
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch,
                dataformats='NHWC')
        if tracker.name == 'wandb':
            tracker.log({phase_name: [wandb.Image(image, caption=
                f'{i}: {args.validation_prompt}') for i, image in enumerate
                (images)]})
    del pipeline
    torch.cuda.empty_cache()
    return images
