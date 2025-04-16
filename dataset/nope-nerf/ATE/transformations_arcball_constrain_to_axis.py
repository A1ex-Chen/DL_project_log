def arcball_constrain_to_axis(point, axis):
    """Return sphere point perpendicular to axis."""
    v = numpy.array(point, dtype=numpy.float64, copy=True)
    a = numpy.array(axis, dtype=numpy.float64, copy=True)
    v -= a * numpy.dot(a, v)
    n = vector_norm(v)
    if n > _EPS:
        if v[2] < 0.0:
            v *= -1.0
        v /= n
        return v
    if a[2] == 1.0:
        return numpy.array([1, 0, 0], dtype=numpy.float64)
    return unit_vector([-a[1], a[0], 0])
