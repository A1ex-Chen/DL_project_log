def next(self, acceleration=0.0):
    """Continue rotation in direction of last drag."""
    q = quaternion_slerp(self._qpre, self._qnow, 2.0 + acceleration, False)
    self._qpre, self._qnow = self._qnow, q
