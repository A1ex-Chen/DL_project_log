def after_val_epoch(self, runner):
    runner.log_buffer.average()
    self.log(runner)
    if self.reset_flag:
        runner.log_buffer.clear_output()
