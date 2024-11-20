def on_batch_begin(self, batch, logs=None):
    if self.seen < self.target:
        self.log_values = []
        self.extra_log_values = []
