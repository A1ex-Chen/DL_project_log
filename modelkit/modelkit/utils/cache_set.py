def set(self, k: bytes, d: Any):
    self.cache.setdefault(k, d)
