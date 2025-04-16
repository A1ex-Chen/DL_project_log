def __getattr__(self, name):
    return getattr(self.optimizer, name)
