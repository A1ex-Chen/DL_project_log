@property
def xywh(self):
    """Get current position in bounding box format (center x, center y, width, height)."""
    ret = np.asarray(self.tlwh).copy()
    ret[:2] += ret[2:] / 2
    return ret
