def rbbox3d_to_corners(rbboxes, origin=[0.5, 0.5, 0.0], axis=2):
    return center_to_corner_box3d(rbboxes[..., :3], rbboxes[..., 3:6],
        rbboxes[..., 6], origin, axis=axis)
