def random_flip(gt_boxes, points, probability=0.5):
    enable = np.random.choice([False, True], replace=False, p=[1 -
        probability, probability])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6] + np.pi
        points[:, 1] = -points[:, 1]
    return gt_boxes, points
