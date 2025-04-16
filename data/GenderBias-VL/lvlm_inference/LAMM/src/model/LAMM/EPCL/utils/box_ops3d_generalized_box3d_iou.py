def generalized_box3d_iou(corners1, corners2, nums_k2=None):
    """
    Input:
        corners1: torch Tensor (B, K1, 8, 3), assume up direction is negative Y
        corners2: torch Tensor (B, K2, 8, 3), assume up direction is negative Y
        mask:
    Returns:
        B x K1 x K2 matrix of generalized IOU
    """
    assert corners1.ndim == 4
    assert corners2.ndim == 4
    assert corners1.shape[0] == corners2.shape[0]
    B, K1, _, _ = corners1.shape
    _, K2, _, _ = corners2.shape
    gious = torch.zeros((B, K1, K2), dtype=torch.float32)
    corners1_np = corners1.detach().cpu().numpy()
    corners2_np = corners2.detach().cpu().numpy()
    for b in range(B):
        for i in range(K1):
            for j in range(K2):
                if nums_k2 is not None and j >= nums_k2[b]:
                    break
                iou, sum_of_vols = box3d_iou(corners1_np[b, i], corners2_np
                    [b, j])
                hull = ConvexHull(np.vstack([corners1_np[b, i], corners2_np
                    [b, j]]))
                C = hull.volume
                giou = iou - (C - sum_of_vols) / C
                gious[b, i, j] = giou
    return gious
