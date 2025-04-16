def on_epoch_begin(self, epoch, logs=None):
    self.epoch_timestamp = datetime.now()
