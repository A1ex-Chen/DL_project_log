def on_train_end(trainer):
    """Logs the best metrics, plots, and confusion matrix at the end of training if DVCLive is active."""
    if live:
        all_metrics = {**trainer.label_loss_items(trainer.tloss, prefix=
            'train'), **trainer.metrics, **trainer.lr}
        for metric, value in all_metrics.items():
            live.log_metric(metric, value, plot=False)
        _log_plots(trainer.plots, 'val')
        _log_plots(trainer.validator.plots, 'val')
        _log_confusion_matrix(trainer.validator)
        if trainer.best.exists():
            live.log_artifact(trainer.best, copy=True, type='model')
        live.end()
