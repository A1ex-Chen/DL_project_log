def __init__(self, initial=None):
    """Initialize virtual trackball control.

        initial : quaternion or rotation matrix

        """
    self._axis = None
    self._axes = None
    self._radius = 1.0
    self._center = [0.0, 0.0]
    self._vdown = numpy.array([0, 0, 1], dtype=numpy.float64)
    self._constrain = False
    if initial is None:
        self._qdown = numpy.array([0, 0, 0, 1], dtype=numpy.float64)
    else:
        initial = numpy.array(initial, dtype=numpy.float64)
        if initial.shape == (4, 4):
            self._qdown = quaternion_from_matrix(initial)
        elif initial.shape == (4,):
            initial /= vector_norm(initial)
            self._qdown = initial
        else:
            raise ValueError('initial not a quaternion or matrix.')
    self._qnow = self._qpre = self._qdown
