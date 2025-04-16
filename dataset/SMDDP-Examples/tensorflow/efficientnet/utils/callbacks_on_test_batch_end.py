def on_test_batch_end(self, batch, logs=None):
    self.global_steps += 1
    self.batch_time.append(time.time() - self.test_begin)
