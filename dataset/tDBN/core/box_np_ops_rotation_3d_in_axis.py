def rotation_3d_in_axis(points, angles, axis=0):
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones,
            zeros], [rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros], [rot_sin, rot_cos,
            zeros], [zeros, zeros, ones]])
    elif axis == 0:
        rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin], [zeros, rot_sin,
            rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError('axis should in range')
    return np.einsum('aij,jka->aik', points, rot_mat_T)
