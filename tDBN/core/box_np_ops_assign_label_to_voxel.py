def assign_label_to_voxel(gt_boxes, coors, voxel_size, coors_range):
    """assign a 0/1 label to each voxel based on whether
    the center of voxel is in gt_box. LIDAR.
    """
    voxel_size = np.array(voxel_size, dtype=gt_boxes.dtype)
    coors_range = np.array(coors_range, dtype=gt_boxes.dtype)
    shift = coors_range[:3]
    voxel_origins = coors[:, ::-1] * voxel_size + shift
    voxel_centers = voxel_origins + voxel_size * 0.5
    gt_box_corners = center_to_corner_box3d(gt_boxes[:, :3] - voxel_size * 
        0.5, gt_boxes[:, 3:6] + voxel_size, gt_boxes[:, 6], origin=[0.5, 
        0.5, 0], axis=2)
    gt_surfaces = corner_to_surfaces_3d(gt_box_corners)
    ret = points_in_convex_polygon_3d_jit(voxel_centers, gt_surfaces)
    return np.any(ret, axis=1).astype(np.int64)
