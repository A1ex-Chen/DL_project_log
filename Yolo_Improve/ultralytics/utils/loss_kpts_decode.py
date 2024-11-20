@staticmethod
def kpts_decode(anchor_points, pred_kpts):
    """Decodes predicted keypoints to image coordinates."""
    y = pred_kpts.clone()
    y[..., :2] *= 2.0
    y[..., 0] += anchor_points[:, [0]] - 0.5
    y[..., 1] += anchor_points[:, [1]] - 0.5
    return y
