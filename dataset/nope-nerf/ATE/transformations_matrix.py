def matrix(self):
    """Return homogeneous rotation matrix."""
    return quaternion_matrix(self._qnow)
