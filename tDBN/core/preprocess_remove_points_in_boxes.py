def remove_points_in_boxes(points, boxes):
    masks = box_np_ops.points_in_rbbox(points, boxes)
    points = points[np.logical_not(masks.any(-1))]
    return points
