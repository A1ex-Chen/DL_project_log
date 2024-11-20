def filter_gt_box_outside_range(gt_boxes, limit_range):
    """remove gtbox outside training range.
    this function should be applied after other prep functions
    Args:
        gt_boxes ([type]): [description]
        limit_range ([type]): [description]
    """
    gt_boxes_bv = box_np_ops.center_to_corner_box2d(gt_boxes[:, [0, 1]],
        gt_boxes[:, [3, 3 + 1]], gt_boxes[:, 6])
    bounding_box = box_np_ops.minmax_to_corner_2d(np.asarray(limit_range)[
        np.newaxis, ...])
    ret = points_in_convex_polygon_jit(gt_boxes_bv.reshape(-1, 2), bounding_box
        )
    return np.any(ret.reshape(-1, 4), axis=1)
