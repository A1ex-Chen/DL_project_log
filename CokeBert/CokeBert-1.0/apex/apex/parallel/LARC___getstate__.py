def __getstate__(self):
    return self.optim.__getstate__()
