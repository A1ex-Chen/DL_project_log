def __enter__(self):
    """Start timing."""
    self.start = self.time()
    return self
