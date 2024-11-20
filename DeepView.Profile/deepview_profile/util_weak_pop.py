def pop(self, key, *args):
    self._dirty_len = True
    return self.data.pop(WeakIdRef(key), *args)
