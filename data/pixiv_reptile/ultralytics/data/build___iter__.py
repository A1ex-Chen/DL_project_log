def __iter__(self):
    """Iterates over the 'sampler' and yields its contents."""
    while True:
        yield from iter(self.sampler)
