def global_rotation(gt_boxes, points, rotation=np.pi / 4):
    if not isinstance(rotation, list):
        rotation = [-rotation, rotation]
    noise_rotation = np.random.uniform(rotation[0], rotation[1])
    points[:, :3] = box_np_ops.rotation_points_single_angle(points[:, :3],
        noise_rotation, axis=2)
    gt_boxes[:, :3] = box_np_ops.rotation_points_single_angle(gt_boxes[:, :
        3], noise_rotation, axis=2)
    gt_boxes[:, 6] += noise_rotation
    return gt_boxes, points
