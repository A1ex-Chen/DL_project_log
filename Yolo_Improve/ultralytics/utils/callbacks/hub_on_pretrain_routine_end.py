def on_pretrain_routine_end(trainer):
    """Logs info before starting timer for upload rate limit."""
    session = getattr(trainer, 'hub_session', None)
    if session:
        session.timers = {'metrics': time(), 'ckpt': time()}
