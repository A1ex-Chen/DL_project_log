def __getattr__(self, attr):
    return getattr(self.terminal, attr)
