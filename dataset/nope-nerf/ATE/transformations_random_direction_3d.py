def random_direction_3d():
    """ equal-area projection according to:
        https://math.stackexchange.com/questions/44689/how-to-find-a-random-axis-or-unit-vector-in-3d
        cfo, 2015/10/16
    """
    z = numpy.random.rand() * 2.0 - 1.0
    t = numpy.random.rand() * 2.0 * numpy.pi
    r = numpy.sqrt(1.0 - z * z)
    x = r * numpy.cos(t)
    y = r * numpy.sin(t)
    return numpy.array([x, y, z], dtype=numpy.float64)
