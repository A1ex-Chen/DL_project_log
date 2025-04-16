def unskew(R):
    """Returns the coordinates of a skew-symmetric matrix
    cfo, 2015/08/13

    """
    return numpy.array([R[2, 1], R[0, 2], R[1, 0]], dtype=numpy.float64)
