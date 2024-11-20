def on_train_epoch_end(self, epoch):
    self.experiment.curr_epoch = epoch
    return
