def mask_points_in_corners(points, box_corners):
    surfaces = box_np_ops.corner_to_surfaces_3d(box_corners)
    mask = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return mask
