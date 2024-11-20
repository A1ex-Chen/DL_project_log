def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    q = numpy.array(quaternion[:4], dtype=numpy.float64, copy=True)
    nq = numpy.dot(q, q)
    if nq < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = numpy.outer(q, q)
    return numpy.array(((1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2
        ] + q[1, 3], 0.0), (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1,
        2] - q[0, 3], 0.0), (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[
        0, 0] - q[1, 1], 0.0), (0.0, 0.0, 0.0, 1.0)), dtype=numpy.float64)
