def __iter__(self):
    """Create a generator that iterates over the data."""
    for item in (self[i] for i in range(len(self))):
        yield item
