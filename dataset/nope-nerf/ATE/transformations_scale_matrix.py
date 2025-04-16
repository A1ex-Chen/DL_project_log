def scale_matrix(factor, origin=None, direction=None):
    """Return matrix to scale by factor around origin in direction.

    Use factor -1 for point symmetry.

    >>> v = (numpy.random.rand(4, 5) - 0.5) * 20.0
    >>> v[3] = 1.0
    >>> S = scale_matrix(-1.234)
    >>> numpy.allclose(numpy.dot(S, v)[:3], -1.234*v[:3])
    True
    >>> factor = random.random() * 10 - 5
    >>> origin = numpy.random.random(3) - 0.5
    >>> direct = numpy.random.random(3) - 0.5
    >>> S = scale_matrix(factor, origin)
    >>> S = scale_matrix(factor, origin, direct)

    """
    if direction is None:
        M = numpy.array(((factor, 0.0, 0.0, 0.0), (0.0, factor, 0.0, 0.0),
            (0.0, 0.0, factor, 0.0), (0.0, 0.0, 0.0, 1.0)), dtype=numpy.float64
            )
        if origin is not None:
            M[:3, 3] = origin[:3]
            M[:3, 3] *= 1.0 - factor
    else:
        direction = unit_vector(direction[:3])
        factor = 1.0 - factor
        M = numpy.identity(4)
        M[:3, :3] -= factor * numpy.outer(direction, direction)
        if origin is not None:
            M[:3, 3] = factor * numpy.dot(origin[:3], direction) * direction
    return M
