def get_momentum(self, epoch=None):
    if epoch is None:
        epoch = self.last_epoch + 1
    return self.lmbd(epoch)
