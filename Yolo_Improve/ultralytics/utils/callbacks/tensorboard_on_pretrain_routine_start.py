def on_pretrain_routine_start(trainer):
    """Initialize TensorBoard logging with SummaryWriter."""
    if SummaryWriter:
        try:
            global WRITER
            WRITER = SummaryWriter(str(trainer.save_dir))
            LOGGER.info(
                f"{PREFIX}Start with 'tensorboard --logdir {trainer.save_dir}', view at http://localhost:6006/"
                )
        except Exception as e:
            LOGGER.warning(
                f'{PREFIX}WARNING ⚠️ TensorBoard not initialized correctly, not logging this run. {e}'
                )
