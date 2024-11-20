def __getstate__(self):
    return self._optimizer.__getstate__()
