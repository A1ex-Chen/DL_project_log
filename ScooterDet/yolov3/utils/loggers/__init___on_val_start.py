def on_val_start(self):
    if self.comet_logger:
        self.comet_logger.on_val_start()
