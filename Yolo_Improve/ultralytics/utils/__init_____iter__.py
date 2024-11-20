def __iter__(self):
    """Return an iterator of key-value pairs from the namespace's attributes."""
    return iter(vars(self).items())
