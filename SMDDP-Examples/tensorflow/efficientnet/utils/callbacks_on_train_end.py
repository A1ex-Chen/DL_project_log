def on_train_end(self, logs=None):
    self.train_finish_time = time.time()
    if self.summary_writer:
        self.summary_writer.flush()
