def __getitem__(self, idx):
    """Return a Results object for a specific index of inference results."""
    return self._apply('__getitem__', idx)
