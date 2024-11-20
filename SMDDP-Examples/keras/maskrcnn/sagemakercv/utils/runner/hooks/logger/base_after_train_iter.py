def after_train_iter(self, runner):
    if self.every_n_inner_iters(runner, self.interval):
        runner.log_buffer.average(self.interval)
    elif self.end_of_epoch(runner) and not self.ignore_last:
        runner.log_buffer.average(self.interval)
    if runner.log_buffer.ready:
        self.log(runner)
        if self.reset_flag:
            runner.log_buffer.clear_output()
