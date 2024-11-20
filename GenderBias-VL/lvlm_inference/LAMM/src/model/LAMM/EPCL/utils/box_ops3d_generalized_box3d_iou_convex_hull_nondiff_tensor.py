def generalized_box3d_iou_convex_hull_nondiff_tensor(corners1: torch.Tensor,
    corners2: torch.Tensor, nums_k2: torch.Tensor, rotated_boxes: bool=True):
    """
    Input:
        corners1: torch Tensor (B, K1, 8, 3), assume up direction is negative Y
        corners2: torch Tensor (B, K2, 8, 3), assume up direction is negative Y
        Assumes that the box is only rotated along Z direction
    Returns:
        B x K1 x K2 matrix of generalized IOU by approximating the boxes to be axis aligned
        The return IOU is differentiable
    """
    assert len(corners1.shape) == 4
    assert len(corners2.shape) == 4
    assert corners1.shape[2] == 8
    assert corners1.shape[3] == 3
    assert corners1.shape[0] == corners2.shape[0]
    assert corners1.shape[2] == corners2.shape[2]
    assert corners1.shape[3] == corners2.shape[3]
    B, K1 = corners1.shape[0], corners1.shape[1]
    _, K2 = corners2.shape[0], corners2.shape[1]
    EPS = 1e-08
    vols1 = box3d_vol_tensor(corners1).clamp(min=EPS)
    vols2 = box3d_vol_tensor(corners2).clamp(min=EPS)
    sum_vols = vols1[:, :, None] + vols2[:, None, :]
    inter_vols = generalized_box3d_iou_tensor_jit(corners1, corners2,
        nums_k2, rotated_boxes, return_inter_vols_only=True)
    enclosing_vols = enclosing_box3d_vol(corners1, corners2)
    if rotated_boxes:
        corners1_np = corners1.detach().cpu().numpy()
        corners2_np = corners2.detach().cpu().numpy()
        mask = inter_vols.detach().cpu().numpy()
        nums_k2 = nums_k2.cpu().numpy()
        enclosing_vols_np = enclosing_vols.detach().cpu().numpy()
        enclosing_vols = enclosing_box3d_convex_hull_numba(corners1_np,
            corners2_np, nums_k2, mask, enclosing_vols_np)
        enclosing_vols = torch.from_numpy(enclosing_vols).to(corners1.device)
    union_vols = (sum_vols - inter_vols).clamp(min=EPS)
    ious = inter_vols / union_vols
    giou_second_term = -(1 - union_vols / enclosing_vols)
    gious = ious + giou_second_term
    good_boxes = (enclosing_vols > 2 * EPS) * (sum_vols > 4 * EPS)
    gious *= good_boxes
    if nums_k2 is not None:
        mask = torch.zeros((B, K1, K2), device=corners1.device, dtype=torch
            .float32)
        for b in range(B):
            mask[b, :, :nums_k2[b]] = 1
        gious *= mask
    return gious
