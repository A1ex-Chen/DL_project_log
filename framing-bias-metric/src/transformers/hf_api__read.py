def _read(self, n=-1):
    self.pbar.update(n)
    return self.read(n)
