def _fetch_trainer_metadata(trainer):
    """Returns metadata for YOLO training including epoch and asset saving status."""
    curr_epoch = trainer.epoch + 1
    train_num_steps_per_epoch = len(trainer.train_loader.dataset
        ) // trainer.batch_size
    curr_step = curr_epoch * train_num_steps_per_epoch
    final_epoch = curr_epoch == trainer.epochs
    save = trainer.args.save
    save_period = trainer.args.save_period
    save_interval = curr_epoch % save_period == 0
    save_assets = (save and save_period > 0 and save_interval and not
        final_epoch)
    return dict(curr_epoch=curr_epoch, curr_step=curr_step, save_assets=
        save_assets, final_epoch=final_epoch)
