def popitem(self):
    self._dirty_len = True
    while True:
        key, value = self.data.popitem()
        o = key()
        if o is not None:
            return o, value
