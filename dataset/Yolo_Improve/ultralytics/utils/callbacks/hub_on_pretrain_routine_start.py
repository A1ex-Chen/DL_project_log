def on_pretrain_routine_start(trainer):
    """Create a remote Ultralytics HUB session to log local model training."""
    if RANK in {-1, 0} and SETTINGS['hub'] is True and SETTINGS['api_key'
        ] and trainer.hub_session is None:
        trainer.hub_session = HUBTrainingSession.create_session(trainer.
            args.model, trainer.args)
