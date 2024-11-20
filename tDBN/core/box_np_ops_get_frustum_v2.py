def get_frustum_v2(bboxes, C, near_clip=0.001, far_clip=100):
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    num_box = bboxes.shape[0]
    z_points = np.array([near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[np
        .newaxis, :, np.newaxis]
    z_points = np.tile(z_points, [num_box, 1, 1])
    box_corners = minmax_to_corner_2d_v2(bboxes)
    near_box_corners = (box_corners - u0v0) / np.array([fku / near_clip, -
        fkv / near_clip], dtype=C.dtype)
    far_box_corners = (box_corners - u0v0) / np.array([fku / far_clip, -fkv /
        far_clip], dtype=C.dtype)
    ret_xy = np.concatenate([near_box_corners, far_box_corners], axis=1)
    ret_xyz = np.concatenate([ret_xy, z_points], axis=-1)
    return ret_xyz
