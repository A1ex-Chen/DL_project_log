def on_train_end(trainer):
    """Perform operations at the end of training."""
    experiment = comet_ml.get_global_experiment()
    if not experiment:
        return
    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata['curr_epoch']
    curr_step = metadata['curr_step']
    plots = trainer.args.plots
    _log_model(experiment, trainer)
    if plots:
        _log_plots(experiment, trainer)
    _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch)
    _log_image_predictions(experiment, trainer.validator, curr_step)
    experiment.end()
    global _comet_image_prediction_count
    _comet_image_prediction_count = 0
