def on_train_epoch_end(self, epoch):
    if self.wandb:
        self.wandb.current_epoch = epoch + 1
    if self.comet_logger:
        self.comet_logger.on_train_epoch_end(epoch)
