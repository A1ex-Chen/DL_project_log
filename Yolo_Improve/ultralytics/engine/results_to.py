def to(self, *args, **kwargs):
    """Moves all tensors in the Results object to the specified device and dtype."""
    return self._apply('to', *args, **kwargs)
