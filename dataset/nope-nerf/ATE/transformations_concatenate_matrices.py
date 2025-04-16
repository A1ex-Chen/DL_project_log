def concatenate_matrices(*matrices):
    """Return concatenation of series of transformation matrices.

    >>> M = numpy.random.rand(16).reshape((4, 4)) - 0.5
    >>> numpy.allclose(M, concatenate_matrices(M))
    True
    >>> numpy.allclose(numpy.dot(M, M.T), concatenate_matrices(M, M.T))
    True

    """
    M = numpy.identity(4)
    for i in matrices:
        M = numpy.dot(M, i)
    return M
