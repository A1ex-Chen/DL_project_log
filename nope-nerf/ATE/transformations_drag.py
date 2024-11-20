def drag(self, point):
    """Update current cursor window coordinates."""
    vnow = arcball_map_to_sphere(point, self._center, self._radius)
    if self._axis is not None:
        vnow = arcball_constrain_to_axis(vnow, self._axis)
    self._qpre = self._qnow
    t = numpy.cross(self._vdown, vnow)
    if numpy.dot(t, t) < _EPS:
        self._qnow = self._qdown
    else:
        q = [t[0], t[1], t[2], numpy.dot(self._vdown, vnow)]
        self._qnow = quaternion_multiply(q, self._qdown)
