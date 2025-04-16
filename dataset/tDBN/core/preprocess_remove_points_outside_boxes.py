def remove_points_outside_boxes(points, boxes):
    masks = box_np_ops.points_in_rbbox(points, boxes)
    points = points[masks.any(-1)]
    return points
