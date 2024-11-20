def on_train_start(trainer):
    """Logs the training parameters if DVCLive logging is active."""
    if live:
        live.log_params(trainer.args)
