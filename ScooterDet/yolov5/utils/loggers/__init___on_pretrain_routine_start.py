def on_pretrain_routine_start(self):
    if self.comet_logger:
        self.comet_logger.on_pretrain_routine_start()
