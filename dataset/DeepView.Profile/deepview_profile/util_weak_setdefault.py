def setdefault(self, key, default=None):
    return self.data.setdefault(WeakIdRef(key, self._remove), default)
