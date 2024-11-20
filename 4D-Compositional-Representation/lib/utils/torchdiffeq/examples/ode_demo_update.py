def update(self, val):
    if self.val is None:
        self.avg = val
    else:
        self.avg = self.avg * self.momentum + val * (1 - self.momentum)
    self.val = val
