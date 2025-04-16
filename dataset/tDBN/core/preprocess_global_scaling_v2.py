def global_scaling_v2(gt_boxes, points, min_scale=0.95, max_scale=1.05):
    noise_scale = np.random.uniform(min_scale, max_scale)
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points
