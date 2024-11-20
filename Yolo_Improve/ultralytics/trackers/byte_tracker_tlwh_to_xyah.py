@staticmethod
def tlwh_to_xyah(tlwh):
    """Convert bounding box to format (center x, center y, aspect ratio, height), where the aspect ratio is width /
        height.
        """
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    ret[2] /= ret[3]
    return ret
