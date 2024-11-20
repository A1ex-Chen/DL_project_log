def on_fit_epoch_end(trainer):
    """Logs training metrics and model info, and advances to next step on the end of each fit epoch."""
    global _training_epoch
    if live and _training_epoch:
        all_metrics = {**trainer.label_loss_items(trainer.tloss, prefix=
            'train'), **trainer.metrics, **trainer.lr}
        for metric, value in all_metrics.items():
            live.log_metric(metric, value)
        if trainer.epoch == 0:
            from ultralytics.utils.torch_utils import model_info_for_loggers
            for metric, value in model_info_for_loggers(trainer).items():
                live.log_metric(metric, value, plot=False)
        _log_plots(trainer.plots, 'train')
        _log_plots(trainer.validator.plots, 'val')
        live.next_step()
        _training_epoch = False
