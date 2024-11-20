def _log_image_predictions(experiment, validator, curr_step):
    """Logs predicted boxes for a single image during training."""
    global _comet_image_prediction_count
    task = validator.args.task
    if task not in COMET_SUPPORTED_TASKS:
        return
    jdict = validator.jdict
    if not jdict:
        return
    predictions_metadata_map = _create_prediction_metadata_map(jdict)
    dataloader = validator.dataloader
    class_label_map = validator.names
    batch_logging_interval = _get_eval_batch_logging_interval()
    max_image_predictions = _get_max_image_predictions_to_log()
    for batch_idx, batch in enumerate(dataloader):
        if (batch_idx + 1) % batch_logging_interval != 0:
            continue
        image_paths = batch['im_file']
        for img_idx, image_path in enumerate(image_paths):
            if _comet_image_prediction_count >= max_image_predictions:
                return
            image_path = Path(image_path)
            annotations = _fetch_annotations(img_idx, image_path, batch,
                predictions_metadata_map, class_label_map)
            _log_images(experiment, [image_path], curr_step, annotations=
                annotations)
            _comet_image_prediction_count += 1
