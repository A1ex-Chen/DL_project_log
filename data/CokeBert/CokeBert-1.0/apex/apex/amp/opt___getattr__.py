def __getattr__(self, attr):
    return getattr(self._optimizer, attr)
