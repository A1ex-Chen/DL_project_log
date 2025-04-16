@master_only
def after_train_epoch(self, runner):
    if not self.every_n_epochs(runner, self.interval):
        return
    if not self.out_dir:
        self.out_dir = runner.work_dir
    checkpoint_dir = os.path.join(self.out_dir, '{:03d}'.format(runner.epoch))
    os.makedirs(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, 'model.h5')
    runner.trainer.model.save_weights(filepath)
    runner.logger.info('Saved checkpoint at: {}'.format(filepath))
