def __next__(self):
    """Return next item in the iterator."""
    if self.count == 1:
        raise StopIteration
    self.count += 1
    return self.paths, self.im0, [''] * self.bs
