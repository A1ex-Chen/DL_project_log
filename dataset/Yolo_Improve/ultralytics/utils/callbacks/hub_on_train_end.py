def on_train_end(trainer):
    """Upload final model and metrics to Ultralytics HUB at the end of training."""
    session = getattr(trainer, 'hub_session', None)
    if session:
        LOGGER.info(f'{PREFIX}Syncing final model...')
        session.upload_model(trainer.epoch, trainer.best, map=trainer.
            metrics.get('metrics/mAP50-95(B)', 0), final=True)
        session.alive = False
        LOGGER.info(
            f'{PREFIX}Done ✅\n{PREFIX}View model at {session.model_url} 🚀')
