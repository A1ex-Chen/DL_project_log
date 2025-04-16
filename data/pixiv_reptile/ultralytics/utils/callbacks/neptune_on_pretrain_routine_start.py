def on_pretrain_routine_start(trainer):
    """Callback function called before the training routine starts."""
    try:
        global run
        run = neptune.init_run(project=trainer.args.project or 'YOLOv8',
            name=trainer.args.name, tags=['YOLOv8'])
        run['Configuration/Hyperparameters'] = {k: ('' if v is None else v) for
            k, v in vars(trainer.args).items()}
    except Exception as e:
        LOGGER.warning(
            f'WARNING ⚠️ NeptuneAI installed but not initialized correctly, not logging this run. {e}'
            )
