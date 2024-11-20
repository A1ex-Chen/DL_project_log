def on_fit_epoch_end(trainer):
    """Callback function called at end of each fit (train+val) epoch."""
    if run and trainer.epoch == 0:
        from ultralytics.utils.torch_utils import model_info_for_loggers
        run['Configuration/Model'] = model_info_for_loggers(trainer)
    _log_scalars(trainer.metrics, trainer.epoch + 1)
