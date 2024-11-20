def update(self, val, n=1):
    self.val = val
    self.sum += val
    self.count += n
    self.avg = self.sum / self.count
