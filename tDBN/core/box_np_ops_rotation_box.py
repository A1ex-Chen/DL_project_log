def rotation_box(box_corners, angle):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angle (float): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T = np.array([[rot_cos, -rot_sin], [rot_sin, rot_cos]], dtype=
        box_corners.dtype)
    return box_corners @ rot_mat_T
