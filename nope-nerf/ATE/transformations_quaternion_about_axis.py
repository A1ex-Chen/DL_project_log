def quaternion_about_axis(angle, axis):
    """Return quaternion for rotation about axis.

    >>> q = quaternion_about_axis(0.123, (1, 0, 0))
    >>> numpy.allclose(q, [0.06146124, 0, 0, 0.99810947])
    True

    """
    quaternion = numpy.zeros((4,), dtype=numpy.float64)
    quaternion[:3] = axis[:3]
    qlen = vector_norm(quaternion)
    if qlen > _EPS:
        quaternion *= math.sin(angle / 2.0) / qlen
    quaternion[3] = math.cos(angle / 2.0)
    return quaternion
