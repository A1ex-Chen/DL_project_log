def superimposition_matrix(v0, v1, scaling=False, usesvd=True):
    """Return matrix to transform given vector set into second vector set.

    v0 and v1 are shape (3, \\*) or (4, \\*) arrays of at least 3 vectors.

    If usesvd is True, the weighted sum of squared deviations (RMSD) is
    minimized according to the algorithm by W. Kabsch [8]. Otherwise the
    quaternion based algorithm by B. Horn [9] is used (slower when using
    this Python implementation).

    The returned matrix performs rotation, translation and uniform scaling
    (if specified).

    >>> v0 = numpy.random.rand(3, 10)
    >>> M = superimposition_matrix(v0, v0)
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> R = random_rotation_matrix(numpy.random.random(3))
    >>> v0 = ((1,0,0), (0,1,0), (0,0,1), (1,1,1))
    >>> v1 = numpy.dot(R, v0)
    >>> M = superimposition_matrix(v0, v1)
    >>> numpy.allclose(v1, numpy.dot(M, v0))
    True
    >>> v0 = (numpy.random.rand(4, 100) - 0.5) * 20.0
    >>> v0[3] = 1.0
    >>> v1 = numpy.dot(R, v0)
    >>> M = superimposition_matrix(v0, v1)
    >>> numpy.allclose(v1, numpy.dot(M, v0))
    True
    >>> S = scale_matrix(random.random())
    >>> T = translation_matrix(numpy.random.random(3)-0.5)
    >>> M = concatenate_matrices(T, R, S)
    >>> v1 = numpy.dot(M, v0)
    >>> v0[:3] += numpy.random.normal(0.0, 1e-9, 300).reshape(3, -1)
    >>> M = superimposition_matrix(v0, v1, scaling=True)
    >>> numpy.allclose(v1, numpy.dot(M, v0))
    True
    >>> M = superimposition_matrix(v0, v1, scaling=True, usesvd=False)
    >>> numpy.allclose(v1, numpy.dot(M, v0))
    True
    >>> v = numpy.empty((4, 100, 3), dtype=numpy.float64)
    >>> v[:, :, 0] = v0
    >>> M = superimposition_matrix(v0, v1, scaling=True, usesvd=False)
    >>> numpy.allclose(v1, numpy.dot(M, v[:, :, 0]))
    True

    """
    v0 = numpy.array(v0, dtype=numpy.float64, copy=False)[:3]
    v1 = numpy.array(v1, dtype=numpy.float64, copy=False)[:3]
    if v0.shape != v1.shape or v0.shape[1] < 3:
        raise ValueError('Vector sets are of wrong shape or type.')
    t0 = numpy.mean(v0, axis=1)
    t1 = numpy.mean(v1, axis=1)
    v0 = v0 - t0.reshape(3, 1)
    v1 = v1 - t1.reshape(3, 1)
    if usesvd:
        u, s, vh = numpy.linalg.svd(numpy.dot(v1, v0.T))
        R = numpy.dot(u, vh)
        if numpy.linalg.det(R) < 0.0:
            R -= numpy.outer(u[:, 2], vh[2, :] * 2.0)
            s[-1] *= -1.0
        M = numpy.identity(4)
        M[:3, :3] = R
    else:
        xx, yy, zz = numpy.sum(v0 * v1, axis=1)
        xy, yz, zx = numpy.sum(v0 * numpy.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = numpy.sum(v0 * numpy.roll(v1, -2, axis=0), axis=1)
        N = (xx + yy + zz, yz - zy, zx - xz, xy - yx), (yz - zy, xx - yy -
            zz, xy + yx, zx + xz), (zx - xz, xy + yx, -xx + yy - zz, yz + zy
            ), (xy - yx, zx + xz, yz + zy, -xx - yy + zz)
        l, V = numpy.linalg.eig(N)
        q = V[:, numpy.argmax(l)]
        q /= vector_norm(q)
        q = numpy.roll(q, -1)
        M = quaternion_matrix(q)
    if scaling:
        v0 *= v0
        v1 *= v1
        M[:3, :3] *= math.sqrt(numpy.sum(v1) / numpy.sum(v0))
    M[:3, 3] = t1
    T = numpy.identity(4)
    T[:3, 3] = -t0
    M = numpy.dot(M, T)
    return M
