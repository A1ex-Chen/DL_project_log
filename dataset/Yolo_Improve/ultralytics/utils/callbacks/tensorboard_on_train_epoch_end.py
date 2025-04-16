def on_train_epoch_end(trainer):
    """Logs scalar statistics at the end of a training epoch."""
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix='train'), 
        trainer.epoch + 1)
    _log_scalars(trainer.lr, trainer.epoch + 1)
