def data_iterator(self, _iter, wrap_around=False):
    """iterates through data and handles wrap around"""
    for i, idx in enumerate(_iter):
        if i < self.wrap_around % self.batch_size:
            continue
        if wrap_around:
            self.wrap_around += 1
            self.wrap_around %= self.batch_size
        yield idx
