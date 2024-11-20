def __getstate__(self):
    return self.optimizer.__getstate__()
