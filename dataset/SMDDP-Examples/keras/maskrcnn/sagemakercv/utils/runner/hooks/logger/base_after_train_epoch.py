def after_train_epoch(self, runner):
    if runner.log_buffer.ready:
        self.log(runner)
        if self.reset_flag:
            runner.log_buffer.clear_output()
