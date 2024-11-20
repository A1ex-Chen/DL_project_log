@master_only
def after_train_iter(self, runner):
    if runner.iter == self.start_iter:
        tf.profiler.experimental.start(runner.tensorboard_dir)
    elif runner.iter == self.stop_iter:
        tf.profiler.experimental.stop()
