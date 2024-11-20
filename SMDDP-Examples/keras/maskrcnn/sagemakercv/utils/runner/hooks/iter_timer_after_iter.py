def after_iter(self, runner):
    if self.every_n_inner_iters(runner, self.interval) and runner.rank == 0:
        iter_time = (time.time() - self.t) / self.interval
        runner.log_buffer.update({'time': iter_time})
        runner.log_buffer.update({'images/s': runner.train_batch_size /
            iter_time})
        self.t = time.time()
