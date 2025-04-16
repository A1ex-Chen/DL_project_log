def on_fit_epoch_end(trainer):
    """Uploads training progress metrics at the end of each epoch."""
    session = getattr(trainer, 'hub_session', None)
    if session:
        all_plots = {**trainer.label_loss_items(trainer.tloss, prefix=
            'train'), **trainer.metrics}
        if trainer.epoch == 0:
            from ultralytics.utils.torch_utils import model_info_for_loggers
            all_plots = {**all_plots, **model_info_for_loggers(trainer)}
        session.metrics_queue[trainer.epoch] = json.dumps(all_plots)
        if session.metrics_upload_failed_queue:
            session.metrics_queue.update(session.metrics_upload_failed_queue)
        if time() - session.timers['metrics'] > session.rate_limits['metrics']:
            session.upload_metrics()
            session.timers['metrics'] = time()
            session.metrics_queue = {}
