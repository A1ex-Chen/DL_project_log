def log_validation(unet, scheduler, args, accelerator, weight_dtype, step,
    name='teacher'):
    logger.info('Running validation... ')
    unet = accelerator.unwrap_model(unet)
    pipeline = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
    pipeline = pipeline.to(device=accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args
            .seed)
    class_labels = [None]
    if args.class_conditional:
        if args.num_classes is not None:
            class_labels = list(range(args.num_classes))
        else:
            logger.warning(
                'The model is class-conditional but the number of classes is not set. The generated images will be unconditional rather than class-conditional.'
                )
    image_logs = []
    for class_label in class_labels:
        images = []
        with torch.autocast('cuda'):
            images = pipeline(num_inference_steps=1, batch_size=args.
                eval_batch_size, class_labels=[class_label] * args.
                eval_batch_size, generator=generator).images
        log = {'images': images}
        if args.class_conditional and class_label is not None:
            log['class_label'] = str(class_label)
        else:
            log['class_label'] = 'images'
        image_logs.append(log)
    for tracker in accelerator.trackers:
        if tracker.name == 'tensorboard':
            for log in image_logs:
                images = log['images']
                class_label = log['class_label']
                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image))
                formatted_images = np.stack(formatted_images)
                tracker.writer.add_images(class_label, formatted_images,
                    step, dataformats='NHWC')
        elif tracker.name == 'wandb':
            formatted_images = []
            for log in image_logs:
                images = log['images']
                class_label = log['class_label']
                for image in images:
                    image = wandb.Image(image, caption=class_label)
                    formatted_images.append(image)
            tracker.log({f'validation/{name}': formatted_images})
        else:
            logger.warning(f'image logging not implemented for {tracker.name}')
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return image_logs
