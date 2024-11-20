def update_batch_loss(self, loss, batch_size):
    self.loss_meter.update(loss, batch_size)
