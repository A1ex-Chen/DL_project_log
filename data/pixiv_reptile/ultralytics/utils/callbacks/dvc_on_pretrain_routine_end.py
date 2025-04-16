def on_pretrain_routine_end(trainer):
    """Logs plots related to the training process at the end of the pretraining routine."""
    _log_plots(trainer.plots, 'train')
