@master_only
def after_train_iter(self, runner):
    if self.every_n_inner_iters(runner, self.interval):
        self.thread_pool.submit(self.images_to_tensorboard, runner)
