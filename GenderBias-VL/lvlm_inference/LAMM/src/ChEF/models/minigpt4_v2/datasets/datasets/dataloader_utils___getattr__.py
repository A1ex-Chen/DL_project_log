def __getattr__(self, name):
    method = self.loader.__getattribute__(name)
    return method
