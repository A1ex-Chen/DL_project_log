def numpy(self):
    """Returns a copy of the Results object with all tensors as numpy arrays."""
    return self._apply('numpy')
