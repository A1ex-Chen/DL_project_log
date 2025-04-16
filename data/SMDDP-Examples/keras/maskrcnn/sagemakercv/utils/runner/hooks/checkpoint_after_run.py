@master_only
def after_run(self, runner):
    if not self.out_dir:
        self.out_dir = runner.work_dir
    checkpoint_dir = os.path.join(self.out_dir, 'trained_model')
    os.makedirs(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, 'model.h5')
    runner.trainer.model.save_weights(filepath)
    runner.logger.info('Saved checkpoint at: {}'.format(filepath))
