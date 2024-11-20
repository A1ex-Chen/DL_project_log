@property
def xyxy(self):
    """Convert bounding box to format (min x, min y, max x, max y), i.e., (top left, bottom right)."""
    ret = self.tlwh.copy()
    ret[2:] += ret[:2]
    return ret
