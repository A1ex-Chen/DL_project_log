def on_epoch_end(self, epoch, logs=None):
    if epoch == 0:
        self.step_per_epoch = self.steps_in_epoch
    epoch_run_time = time.time() - self.epoch_start
    self.epoch_runtime_log.append(epoch_run_time)
    self.steps_before_epoch += self.steps_in_epoch
    self.steps_in_epoch = 0
