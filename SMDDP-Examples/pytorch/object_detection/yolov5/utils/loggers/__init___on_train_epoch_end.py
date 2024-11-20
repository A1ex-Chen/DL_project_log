def on_train_epoch_end(self, epoch):
    if self.wandb:
        self.wandb.current_epoch = epoch + 1
