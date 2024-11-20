def quaternion_conjugate(quaternion):
    """Return conjugate of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quaternion_conjugate(q0)
    >>> q1[3] == q0[3] and all(q1[:3] == -q0[:3])
    True

    """
    return numpy.array((-quaternion[0], -quaternion[1], -quaternion[2],
        quaternion[3]), dtype=numpy.float64)
