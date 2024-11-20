def _log_images(experiment, image_paths, curr_step, annotations=None):
    """Logs images to the experiment with optional annotations."""
    if annotations:
        for image_path, annotation in zip(image_paths, annotations):
            experiment.log_image(image_path, name=image_path.stem, step=
                curr_step, annotations=annotation)
    else:
        for image_path in image_paths:
            experiment.log_image(image_path, name=image_path.stem, step=
                curr_step)
