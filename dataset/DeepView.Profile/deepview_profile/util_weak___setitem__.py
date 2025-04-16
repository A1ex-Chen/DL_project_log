def __setitem__(self, key, value):
    self.data[WeakIdRef(key, self._remove)] = value
