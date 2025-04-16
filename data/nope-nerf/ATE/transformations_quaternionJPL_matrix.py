def quaternionJPL_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion in JPL notation.
       quaternion = [x y z w]
    """
    q0 = quaternion[0]
    q1 = quaternion[1]
    q2 = quaternion[2]
    q3 = quaternion[3]
    return numpy.array([[q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2, 2.0 * q0 *
        q1 + 2.0 * q2 * q3, 2.0 * q0 * q2 - 2.0 * q1 * q3, 0], [2.0 * q0 *
        q1 - 2.0 * q2 * q3, -q0 ** 2 + q1 ** 2 - q2 ** 2 + q3 ** 2, 2.0 *
        q0 * q3 + 2.0 * q1 * q2, 0], [2.0 * q0 * q2 + 2.0 * q1 * q3, 2.0 *
        q1 * q2 - 2.0 * q0 * q3, -q0 ** 2 - q1 ** 2 + q2 ** 2 + q3 ** 2, 0],
        [0, 0, 0, 1.0]], dtype=numpy.float64)
