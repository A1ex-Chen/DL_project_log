def on_model_save(trainer):
    """Saves checkpoints to Ultralytics HUB with rate limiting."""
    session = getattr(trainer, 'hub_session', None)
    if session:
        is_best = trainer.best_fitness == trainer.fitness
        if time() - session.timers['ckpt'] > session.rate_limits['ckpt']:
            LOGGER.info(
                f'{PREFIX}Uploading checkpoint {HUB_WEB_ROOT}/models/{session.model.id}'
                )
            session.upload_model(trainer.epoch, trainer.last, is_best)
            session.timers['ckpt'] = time()
