def __enter__(self):
    self.start = self.time()
    return self
