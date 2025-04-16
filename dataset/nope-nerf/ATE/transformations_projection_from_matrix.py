def projection_from_matrix(matrix, pseudo=False):
    """Return projection plane and perspective point from projection matrix.

    Return values are same as arguments for projection_matrix function:
    point, normal, direction, perspective, and pseudo.

    >>> point = numpy.random.random(3) - 0.5
    >>> normal = numpy.random.random(3) - 0.5
    >>> direct = numpy.random.random(3) - 0.5
    >>> persp = numpy.random.random(3) - 0.5
    >>> P0 = projection_matrix(point, normal)
    >>> result = projection_from_matrix(P0)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True
    >>> P0 = projection_matrix(point, normal, direct)
    >>> result = projection_from_matrix(P0)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True
    >>> P0 = projection_matrix(point, normal, perspective=persp, pseudo=False)
    >>> result = projection_from_matrix(P0, pseudo=False)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True
    >>> P0 = projection_matrix(point, normal, perspective=persp, pseudo=True)
    >>> result = projection_from_matrix(P0, pseudo=True)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)
    M33 = M[:3, :3]
    l, V = numpy.linalg.eig(M)
    i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-08)[0]
    if not pseudo and len(i):
        point = numpy.real(V[:, i[-1]]).squeeze()
        point /= point[3]
        l, V = numpy.linalg.eig(M33)
        i = numpy.where(abs(numpy.real(l)) < 1e-08)[0]
        if not len(i):
            raise ValueError('no eigenvector corresponding to eigenvalue 0')
        direction = numpy.real(V[:, i[0]]).squeeze()
        direction /= vector_norm(direction)
        l, V = numpy.linalg.eig(M33.T)
        i = numpy.where(abs(numpy.real(l)) < 1e-08)[0]
        if len(i):
            normal = numpy.real(V[:, i[0]]).squeeze()
            normal /= vector_norm(normal)
            return point, normal, direction, None, False
        else:
            return point, direction, None, None, False
    else:
        i = numpy.where(abs(numpy.real(l)) > 1e-08)[0]
        if not len(i):
            raise ValueError('no eigenvector not corresponding to eigenvalue 0'
                )
        point = numpy.real(V[:, i[-1]]).squeeze()
        point /= point[3]
        normal = -M[3, :3]
        perspective = M[:3, 3] / numpy.dot(point[:3], normal)
        if pseudo:
            perspective -= normal
        return point, normal, None, perspective, pseudo
