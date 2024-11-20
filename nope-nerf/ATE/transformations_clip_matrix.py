def clip_matrix(left, right, bottom, top, near, far, perspective=False):
    """Return matrix to obtain normalized device coordinates from frustrum.

    The frustrum bounds are axis-aligned along x (left, right),
    y (bottom, top) and z (near, far).

    Normalized device coordinates are in range [-1, 1] if coordinates are
    inside the frustrum.

    If perspective is True the frustrum is a truncated pyramid with the
    perspective point at origin and direction along z axis, otherwise an
    orthographic canonical view volume (a box).

    Homogeneous coordinates transformed by the perspective clip matrix
    need to be dehomogenized (devided by w coordinate).

    >>> frustrum = numpy.random.rand(6)
    >>> frustrum[1] += frustrum[0]
    >>> frustrum[3] += frustrum[2]
    >>> frustrum[5] += frustrum[4]
    >>> M = clip_matrix(*frustrum, perspective=False)
    >>> numpy.dot(M, [frustrum[0], frustrum[2], frustrum[4], 1.0])
    array([-1., -1., -1.,  1.])
    >>> numpy.dot(M, [frustrum[1], frustrum[3], frustrum[5], 1.0])
    array([ 1.,  1.,  1.,  1.])
    >>> M = clip_matrix(*frustrum, perspective=True)
    >>> v = numpy.dot(M, [frustrum[0], frustrum[2], frustrum[4], 1.0])
    >>> v / v[3]
    array([-1., -1., -1.,  1.])
    >>> v = numpy.dot(M, [frustrum[1], frustrum[3], frustrum[4], 1.0])
    >>> v / v[3]
    array([ 1.,  1., -1.,  1.])

    """
    if left >= right or bottom >= top or near >= far:
        raise ValueError('invalid frustrum')
    if perspective:
        if near <= _EPS:
            raise ValueError('invalid frustrum: near <= 0')
        t = 2.0 * near
        M = (-t / (right - left), 0.0, (right + left) / (right - left), 0.0), (
            0.0, -t / (top - bottom), (top + bottom) / (top - bottom), 0.0), (
            0.0, 0.0, -(far + near) / (far - near), t * far / (far - near)), (
            0.0, 0.0, -1.0, 0.0)
    else:
        M = (2.0 / (right - left), 0.0, 0.0, (right + left) / (left - right)
            ), (0.0, 2.0 / (top - bottom), 0.0, (top + bottom) / (bottom - top)
            ), (0.0, 0.0, 2.0 / (far - near), (far + near) / (near - far)), (
            0.0, 0.0, 0.0, 1.0)
    return numpy.array(M, dtype=numpy.float64)
