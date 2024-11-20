def filter_gt_low_points(gt_boxes, points, num_gt_points, point_num_threshold=2
    ):
    points_mask = np.ones([points.shape[0]], np.bool)
    gt_boxes_mask = np.ones([gt_boxes.shape[0]], np.bool)
    for i, num in enumerate(num_gt_points):
        if num <= point_num_threshold:
            masks = box_np_ops.points_in_rbbox(points, gt_boxes[i:i + 1])
            masks = masks.reshape([-1])
            points_mask &= np.logical_not(masks)
            gt_boxes_mask[i] = False
    return gt_boxes[gt_boxes_mask], points[points_mask]
