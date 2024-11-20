def step(self, epoch=None):
    if epoch is None:
        epoch = self.last_epoch + 1
    self.last_epoch = epoch
    self.model.apply(self.setter(self.lmbd(epoch)))
