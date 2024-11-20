@property
def tlwh(self):
    """Get current position in bounding box format `(top left x, top left y, width, height)`."""
    if self.mean is None:
        return self._tlwh.copy()
    ret = self.mean[:4].copy()
    ret[:2] -= ret[2:] / 2
    return ret
