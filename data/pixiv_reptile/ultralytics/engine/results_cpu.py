def cpu(self):
    """Returns a copy of the Results object with all its tensors moved to CPU memory."""
    return self._apply('cpu')
