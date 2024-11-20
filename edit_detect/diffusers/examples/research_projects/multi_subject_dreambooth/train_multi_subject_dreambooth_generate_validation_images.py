def generate_validation_images(text_encoder: Module, tokenizer: Module,
    unet: Module, vae: Module, arguments: argparse.Namespace, accelerator:
    Accelerator, weight_dtype: dtype):
    logger.info('Running validation images.')
    pipeline_args = {}
    if text_encoder is not None:
        pipeline_args['text_encoder'] = accelerator.unwrap_model(text_encoder)
    if vae is not None:
        pipeline_args['vae'] = vae
    pipeline = DiffusionPipeline.from_pretrained(arguments.
        pretrained_model_name_or_path, tokenizer=tokenizer, unet=
        accelerator.unwrap_model(unet), revision=arguments.revision,
        torch_dtype=weight_dtype, **pipeline_args)
    scheduler_args = {}
    if 'variance_type' in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type
        if variance_type in ['learned', 'learned_range']:
            variance_type = 'fixed_small'
        scheduler_args['variance_type'] = variance_type
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.
        scheduler.config, **scheduler_args)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = None if arguments.seed is None else torch.Generator(device=
        accelerator.device).manual_seed(arguments.seed)
    images_sets = []
    for vp, nvi, vnp, vis, vgs in zip(arguments.validation_prompt,
        arguments.validation_number_images, arguments.
        validation_negative_prompt, arguments.validation_inference_steps,
        arguments.validation_guidance_scale):
        images = []
        if vp is not None:
            logger.info(
                f"Generating {nvi} images with prompt: '{vp}', negative prompt: '{vnp}', inference steps: {vis}, guidance scale: {vgs}."
                )
            pipeline_args = {'prompt': vp, 'negative_prompt': vnp,
                'num_inference_steps': vis, 'guidance_scale': vgs}
            for _ in range(nvi):
                with torch.autocast('cuda'):
                    image = pipeline(**pipeline_args, num_images_per_prompt
                        =1, generator=generator).images[0]
                images.append(image)
        images_sets.append(images)
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return images_sets
