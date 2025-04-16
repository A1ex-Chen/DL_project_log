def on_train_start(self):
    if self.comet_logger:
        self.comet_logger.on_train_start()
