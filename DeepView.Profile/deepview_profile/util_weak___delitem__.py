def __delitem__(self, key):
    self._dirty_len = True
    del self.data[WeakIdRef(key)]
