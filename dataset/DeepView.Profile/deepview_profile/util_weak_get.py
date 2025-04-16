def get(self, key, default=None):
    return self.data.get(WeakIdRef(key), default)
