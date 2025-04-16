def place(self, center, radius):
    """Place Arcball, e.g. when window size changes.

        center : sequence[2]
            Window coordinates of trackball center.
        radius : float
            Radius of trackball in window coordinates.

        """
    self._radius = float(radius)
    self._center[0] = center[0]
    self._center[1] = center[1]
