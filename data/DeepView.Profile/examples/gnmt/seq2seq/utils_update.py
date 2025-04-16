def update(self, val, n=1):
    self.val = val
    if self.skip:
        self.skip = False
    else:
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
