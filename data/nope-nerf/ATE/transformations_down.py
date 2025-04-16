def down(self, point):
    """Set initial cursor window coordinates and pick constrain-axis."""
    self._vdown = arcball_map_to_sphere(point, self._center, self._radius)
    self._qdown = self._qpre = self._qnow
    if self._constrain and self._axes is not None:
        self._axis = arcball_nearest_axis(self._vdown, self._axes)
        self._vdown = arcball_constrain_to_axis(self._vdown, self._axis)
    else:
        self._axis = None
