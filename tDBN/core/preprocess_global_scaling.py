def global_scaling(gt_boxes, points, scale=0.05):
    if not isinstance(scale, list):
        scale = [-scale, scale]
    noise_scale = np.random.uniform(scale[0] + 1, scale[1] + 1)
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points
