def log_validation_images_to_tracker(images: List[np.array], label: str,
    validation_prompt: str, accelerator: Accelerator, epoch: int):
    logger.info(
        f'Logging images to tracker for validation prompt: {validation_prompt}.'
        )
    for tracker in accelerator.trackers:
        if tracker.name == 'tensorboard':
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images('validation', np_images, epoch,
                dataformats='NHWC')
        if tracker.name == 'wandb':
            tracker.log({'validation': [wandb.Image(image, caption=
                f'{label}_{epoch}_{i}: {validation_prompt}') for i, image in
                enumerate(images)]})
