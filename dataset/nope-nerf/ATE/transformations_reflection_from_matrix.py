def reflection_from_matrix(matrix):
    """Return mirror plane point and normal vector from reflection matrix.

    >>> v0 = numpy.random.random(3) - 0.5
    >>> v1 = numpy.random.random(3) - 0.5
    >>> M0 = reflection_matrix(v0, v1)
    >>> point, normal = reflection_from_matrix(M0)
    >>> M1 = reflection_matrix(point, normal)
    >>> is_same_transform(M0, M1)
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)
    l, V = numpy.linalg.eig(M[:3, :3])
    i = numpy.where(abs(numpy.real(l) + 1.0) < 1e-08)[0]
    if not len(i):
        raise ValueError('no unit eigenvector corresponding to eigenvalue -1')
    normal = numpy.real(V[:, i[0]]).squeeze()
    l, V = numpy.linalg.eig(M)
    i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-08)[0]
    if not len(i):
        raise ValueError('no unit eigenvector corresponding to eigenvalue 1')
    point = numpy.real(V[:, i[-1]]).squeeze()
    point /= point[3]
    return point, normal
