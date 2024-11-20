def on_batch_begin(self, batch, logs=None):
    if not self.start_time:
        self.start_time = time.time()
    if not self.timestamp_log:
        self.timestamp_log.append(BatchTimestamp(self.global_steps, self.
            start_time))
