def points_in_rbbox(points, rbbox, lidar=True):
    if lidar:
        h_axis = 2
        origin = [0.5, 0.5, 0]
    else:
        origin = [0.5, 1.0, 0.5]
        h_axis = 1
    rbbox_corners = center_to_corner_box3d(rbbox[:, :3], rbbox[:, 3:6],
        rbbox[:, 6], origin=origin, axis=h_axis)
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices
