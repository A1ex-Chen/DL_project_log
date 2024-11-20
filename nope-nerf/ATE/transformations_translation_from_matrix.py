def translation_from_matrix(matrix):
    """Return translation vector from translation matrix.

    >>> v0 = numpy.random.random(3) - 0.5
    >>> v1 = translation_from_matrix(translation_matrix(v0))
    >>> numpy.allclose(v0, v1)
    True

    """
    return numpy.array(matrix, copy=False)[:3, 3].copy()
