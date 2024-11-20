def rbbox3d_to_bev_corners(rbboxes, origin=0.5):
    return center_to_corner_box2d(rbboxes[..., :2], rbboxes[..., 3:5],
        rbboxes[..., 6], origin)
