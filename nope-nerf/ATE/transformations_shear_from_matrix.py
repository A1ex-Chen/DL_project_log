def shear_from_matrix(matrix):
    """Return shear angle, direction and plane from shear matrix.

    >>> angle = (random.random() - 0.5) * 4*math.pi
    >>> direct = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> normal = numpy.cross(direct, numpy.random.random(3))
    >>> S0 = shear_matrix(angle, direct, point, normal)
    >>> angle, direct, point, normal = shear_from_matrix(S0)
    >>> S1 = shear_matrix(angle, direct, point, normal)
    >>> is_same_transform(S0, S1)
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)
    M33 = M[:3, :3]
    l, V = numpy.linalg.eig(M33)
    i = numpy.where(abs(numpy.real(l) - 1.0) < 0.0001)[0]
    if len(i) < 2:
        raise ValueError('No two linear independent eigenvectors found %s' % l)
    V = numpy.real(V[:, i]).squeeze().T
    lenorm = -1.0
    for i0, i1 in ((0, 1), (0, 2), (1, 2)):
        n = numpy.cross(V[i0], V[i1])
        l = vector_norm(n)
        if l > lenorm:
            lenorm = l
            normal = n
    normal /= lenorm
    direction = numpy.dot(M33 - numpy.identity(3), normal)
    angle = vector_norm(direction)
    direction /= angle
    angle = math.atan(angle)
    l, V = numpy.linalg.eig(M)
    i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-08)[0]
    if not len(i):
        raise ValueError('no eigenvector corresponding to eigenvalue 1')
    point = numpy.real(V[:, i[-1]]).squeeze()
    point /= point[3]
    return angle, direction, point, normal
