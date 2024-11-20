def on_fit_epoch_end(self, result, epoch):
    self.log_metrics(result, epoch=epoch)
