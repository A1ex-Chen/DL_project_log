@staticmethod
def tlwh_to_xywh(tlwh):
    """Convert bounding box to format `(center x, center y, width, height)`."""
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    return ret
