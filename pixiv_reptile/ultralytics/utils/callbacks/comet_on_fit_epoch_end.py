def on_fit_epoch_end(trainer):
    """Logs model assets at the end of each epoch."""
    experiment = comet_ml.get_global_experiment()
    if not experiment:
        return
    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata['curr_epoch']
    curr_step = metadata['curr_step']
    save_assets = metadata['save_assets']
    experiment.log_metrics(trainer.metrics, step=curr_step, epoch=curr_epoch)
    experiment.log_metrics(trainer.lr, step=curr_step, epoch=curr_epoch)
    if curr_epoch == 1:
        from ultralytics.utils.torch_utils import model_info_for_loggers
        experiment.log_metrics(model_info_for_loggers(trainer), step=
            curr_step, epoch=curr_epoch)
    if not save_assets:
        return
    _log_model(experiment, trainer)
    if _should_log_confusion_matrix():
        _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch)
    if _should_log_image_predictions():
        _log_image_predictions(experiment, trainer.validator, curr_step)
