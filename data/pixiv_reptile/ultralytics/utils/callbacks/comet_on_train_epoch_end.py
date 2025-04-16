def on_train_epoch_end(trainer):
    """Log metrics and save batch images at the end of training epochs."""
    experiment = comet_ml.get_global_experiment()
    if not experiment:
        return
    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata['curr_epoch']
    curr_step = metadata['curr_step']
    experiment.log_metrics(trainer.label_loss_items(trainer.tloss, prefix=
        'train'), step=curr_step, epoch=curr_epoch)
    if curr_epoch == 1:
        _log_images(experiment, trainer.save_dir.glob('train_batch*.jpg'),
            curr_step)
