def on_train_epoch_end(trainer):
    """Callback function called at end of each training epoch."""
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix='train'), 
        trainer.epoch + 1)
    _log_scalars(trainer.lr, trainer.epoch + 1)
    if trainer.epoch == 1:
        _log_images({f.stem: str(f) for f in trainer.save_dir.glob(
            'train_batch*.jpg')}, 'Mosaic')
