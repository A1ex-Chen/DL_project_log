def cuda(self):
    """Moves all tensors in the Results object to GPU memory."""
    return self._apply('cuda')
