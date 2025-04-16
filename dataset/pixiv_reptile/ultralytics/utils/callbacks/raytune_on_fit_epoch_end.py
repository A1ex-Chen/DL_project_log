def on_fit_epoch_end(trainer):
    """Sends training metrics to Ray Tune at end of each epoch."""
    if ray.train._internal.session._get_session():
        metrics = trainer.metrics
        metrics['epoch'] = trainer.epoch
        session.report(metrics)
