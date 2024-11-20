def on_train_epoch_end(trainer):
    """Logs debug samples for the first epoch of YOLO training and report current training progress."""
    if (task := Task.current_task()):
        if trainer.epoch == 1:
            _log_debug_samples(sorted(trainer.save_dir.glob(
                'train_batch*.jpg')), 'Mosaic')
        for k, v in trainer.label_loss_items(trainer.tloss, prefix='train'
            ).items():
            task.get_logger().report_scalar('train', k, v, iteration=
                trainer.epoch)
        for k, v in trainer.lr.items():
            task.get_logger().report_scalar('lr', k, v, iteration=trainer.epoch
                )
