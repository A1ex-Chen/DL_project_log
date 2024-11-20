def filter_gt_box_outside_range_by_center(gt_boxes, limit_range):
    """remove gtbox outside training range.
    this function should be applied after other prep functions
    Args:
        gt_boxes ([type]): [description]
        limit_range ([type]): [description]
    """
    gt_box_centers = gt_boxes[:, :2]
    bounding_box = box_np_ops.minmax_to_corner_2d(np.asarray(limit_range)[
        np.newaxis, ...])
    ret = points_in_convex_polygon_jit(gt_box_centers, bounding_box)
    return ret.reshape(-1)
