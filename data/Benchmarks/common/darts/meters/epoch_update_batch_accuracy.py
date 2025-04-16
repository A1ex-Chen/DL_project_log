def update_batch_accuracy(self, acc, batch_size):
    self.acc_meter.update(acc, batch_size)
