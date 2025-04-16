def scale_from_matrix(matrix):
    """Return scaling factor, origin and direction from scaling matrix.

    >>> factor = random.random() * 10 - 5
    >>> origin = numpy.random.random(3) - 0.5
    >>> direct = numpy.random.random(3) - 0.5
    >>> S0 = scale_matrix(factor, origin)
    >>> factor, origin, direction = scale_from_matrix(S0)
    >>> S1 = scale_matrix(factor, origin, direction)
    >>> is_same_transform(S0, S1)
    True
    >>> S0 = scale_matrix(factor, origin, direct)
    >>> factor, origin, direction = scale_from_matrix(S0)
    >>> S1 = scale_matrix(factor, origin, direction)
    >>> is_same_transform(S0, S1)
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)
    M33 = M[:3, :3]
    factor = numpy.trace(M33) - 2.0
    try:
        l, V = numpy.linalg.eig(M33)
        i = numpy.where(abs(numpy.real(l) - factor) < 1e-08)[0][0]
        direction = numpy.real(V[:, i]).squeeze()
        direction /= vector_norm(direction)
    except IndexError:
        factor = (factor + 2.0) / 3.0
        direction = None
    l, V = numpy.linalg.eig(M)
    i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-08)[0]
    if not len(i):
        raise ValueError('no eigenvector corresponding to eigenvalue 1')
    origin = numpy.real(V[:, i[-1]]).squeeze()
    origin /= origin[3]
    return factor, origin, direction
