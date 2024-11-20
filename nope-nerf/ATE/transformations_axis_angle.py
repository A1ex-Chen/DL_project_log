def axis_angle(axis, theta):
    """Compute a rotation matrix from an axis and an angle.
    Returns 3x3 Matrix.
    Is the same as transformations.rotation_matrix(theta, axis).
    cfo, 2015/08/13

    """
    if theta * theta > _EPS:
        wx = axis[0]
        wy = axis[1]
        wz = axis[2]
        costheta = numpy.cos(theta)
        sintheta = numpy.sin(theta)
        c_1 = 1.0 - costheta
        wx_sintheta = wx * sintheta
        wy_sintheta = wy * sintheta
        wz_sintheta = wz * sintheta
        C00 = c_1 * wx * wx
        C01 = c_1 * wx * wy
        C02 = c_1 * wx * wz
        C11 = c_1 * wy * wy
        C12 = c_1 * wy * wz
        C22 = c_1 * wz * wz
        R = numpy.zeros((3, 3), dtype=numpy.float64)
        R[0, 0] = costheta + C00
        R[1, 0] = wz_sintheta + C01
        R[2, 0] = -wy_sintheta + C02
        R[0, 1] = -wz_sintheta + C01
        R[1, 1] = costheta + C11
        R[2, 1] = wx_sintheta + C12
        R[0, 2] = wy_sintheta + C02
        R[1, 2] = -wx_sintheta + C12
        R[2, 2] = costheta + C22
        return R
    else:
        return first_order_rotation(axis * theta)
