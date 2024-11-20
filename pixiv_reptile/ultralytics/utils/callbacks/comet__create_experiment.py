def _create_experiment(args):
    """Ensures that the experiment object is only created in a single process during distributed training."""
    if RANK not in {-1, 0}:
        return
    try:
        comet_mode = _get_comet_mode()
        _project_name = os.getenv('COMET_PROJECT_NAME', args.project)
        experiment = _get_experiment_type(comet_mode, _project_name)
        experiment.log_parameters(vars(args))
        experiment.log_others({'eval_batch_logging_interval':
            _get_eval_batch_logging_interval(),
            'log_confusion_matrix_on_eval': _should_log_confusion_matrix(),
            'log_image_predictions': _should_log_image_predictions(),
            'max_image_predictions': _get_max_image_predictions_to_log()})
        experiment.log_other('Created from', 'yolov8')
    except Exception as e:
        LOGGER.warning(
            f'WARNING ⚠️ Comet installed but not initialized correctly, not logging this run. {e}'
            )
