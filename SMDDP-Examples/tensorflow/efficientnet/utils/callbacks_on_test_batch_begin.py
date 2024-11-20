def on_test_batch_begin(self, epoch, logs=None):
    self.test_begin = time.time()
