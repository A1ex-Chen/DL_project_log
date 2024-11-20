def log_validation(text_encoder, tokenizer, unet, vae, args, accelerator,
    weight_dtype, global_step, prompt_embeds, negative_prompt_embeds):
    logger.info(
        f"""Running validation... 
 Generating {args.num_validation_images} images with prompt: {args.validation_prompt}."""
        )
    pipeline_args = {}
    if vae is not None:
        pipeline_args['vae'] = vae
    pipeline = DiffusionPipeline.from_pretrained(args.
        pretrained_model_name_or_path, tokenizer=tokenizer, text_encoder=
        text_encoder, unet=unet, revision=args.revision, variant=args.
        variant, torch_dtype=weight_dtype, **pipeline_args)
    scheduler_args = {}
    if 'variance_type' in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type
        if variance_type in ['learned', 'learned_range']:
            variance_type = 'fixed_small'
        scheduler_args['variance_type'] = variance_type
    module = importlib.import_module('diffusers')
    scheduler_class = getattr(module, args.validation_scheduler)
    pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.
        config, **scheduler_args)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    if args.pre_compute_text_embeddings:
        pipeline_args = {'prompt_embeds': prompt_embeds,
            'negative_prompt_embeds': negative_prompt_embeds}
    else:
        pipeline_args = {'prompt': args.validation_prompt}
    generator = None if args.seed is None else torch.Generator(device=
        accelerator.device).manual_seed(args.seed)
    images = []
    if args.validation_images is None:
        for _ in range(args.num_validation_images):
            with torch.autocast('cuda'):
                image = pipeline(**pipeline_args, num_inference_steps=25,
                    generator=generator).images[0]
            images.append(image)
    else:
        for image in args.validation_images:
            image = Image.open(image)
            image = pipeline(**pipeline_args, image=image, generator=generator
                ).images[0]
            images.append(image)
    for tracker in accelerator.trackers:
        if tracker.name == 'tensorboard':
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images('validation', np_images, global_step,
                dataformats='NHWC')
        if tracker.name == 'wandb':
            tracker.log({'validation': [wandb.Image(image, caption=
                f'{i}: {args.validation_prompt}') for i, image in enumerate
                (images)]})
    del pipeline
    torch.cuda.empty_cache()
    return images
