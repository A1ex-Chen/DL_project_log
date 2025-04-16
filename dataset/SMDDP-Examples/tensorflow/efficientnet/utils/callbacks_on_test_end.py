def on_test_end(self, epoch, logs=None):
    self.eval_time = sum(self.batch_time) - self.batch_time[0]
