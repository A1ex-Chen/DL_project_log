def random_quaternion(rand=None):
    """Return uniform random unit quaternion.

    rand: array like or None
        Three independent random variables that are uniformly distributed
        between 0 and 1.

    >>> q = random_quaternion()
    >>> numpy.allclose(1.0, vector_norm(q))
    True
    >>> q = random_quaternion(numpy.random.random(3))
    >>> q.shape
    (4,)

    """
    if rand is None:
        rand = numpy.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = numpy.sqrt(1.0 - rand[0])
    r2 = numpy.sqrt(rand[0])
    pi2 = math.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return numpy.array((numpy.sin(t1) * r1, numpy.cos(t1) * r1, numpy.sin(
        t2) * r2, numpy.cos(t2) * r2), dtype=numpy.float64)
