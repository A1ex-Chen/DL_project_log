def first_order_rotation(rotvec):
    """First order approximation of a rotation: I + skew(rotvec)
    cfo, 2015/08/13

    """
    R = numpy.zeros((3, 3), dtype=numpy.float64)
    R[0, 0] = 1.0
    R[1, 0] = rotvec[2]
    R[2, 0] = -rotvec[1]
    R[0, 1] = -rotvec[2]
    R[1, 1] = 1.0
    R[2, 1] = rotvec[0]
    R[0, 2] = rotvec[1]
    R[1, 2] = -rotvec[0]
    R[2, 2] = 1.0
    return R
