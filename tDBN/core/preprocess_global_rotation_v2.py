def global_rotation_v2(gt_boxes, points, min_rad=-np.pi / 4, max_rad=np.pi / 4
    ):
    noise_rotation = np.random.uniform(min_rad, max_rad)
    points[:, :3] = box_np_ops.rotation_points_single_angle(points[:, :3],
        noise_rotation, axis=2)
    gt_boxes[:, :3] = box_np_ops.rotation_points_single_angle(gt_boxes[:, :
        3], noise_rotation, axis=2)
    gt_boxes[:, 6] += noise_rotation
    return gt_boxes, points
