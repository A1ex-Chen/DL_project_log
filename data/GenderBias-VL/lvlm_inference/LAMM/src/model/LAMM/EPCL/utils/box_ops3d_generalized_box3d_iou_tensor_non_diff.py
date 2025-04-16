def generalized_box3d_iou_tensor_non_diff(corners1: torch.Tensor, corners2:
    torch.Tensor, nums_k2: torch.Tensor, rotated_boxes: bool=True,
    return_inter_vols_only: bool=False, approximate: bool=True):
    if batch_intersect is None:
        return generalized_box3d_iou_tensor_jit(corners1, corners2, nums_k2,
            rotated_boxes, return_inter_vols_only)
    else:
        assert len(corners1.shape) == 4
        assert len(corners2.shape) == 4
        assert corners1.shape[2] == 8
        assert corners1.shape[3] == 3
        assert corners1.shape[0] == corners2.shape[0]
        assert corners1.shape[2] == corners2.shape[2]
        assert corners1.shape[3] == corners2.shape[3]
        B, K1 = corners1.shape[0], corners1.shape[1]
        _, K2 = corners2.shape[0], corners2.shape[1]
        ymax = torch.min(corners1[:, :, 0, 1][:, :, None], corners2[:, :, 0,
            1][:, None, :])
        ymin = torch.max(corners1[:, :, 4, 1][:, :, None], corners2[:, :, 4,
            1][:, None, :])
        height = (ymax - ymin).clamp(min=0)
        EPS = 1e-08
        idx = torch.arange(start=3, end=-1, step=-1, device=corners1.device)
        idx2 = torch.tensor([0, 2], dtype=torch.int64, device=corners1.device)
        rect1 = corners1[:, :, idx, :]
        rect2 = corners2[:, :, idx, :]
        rect1 = rect1[:, :, :, idx2]
        rect2 = rect2[:, :, :, idx2]
        lt = torch.max(rect1[:, :, 1][:, :, None, :], rect2[:, :, 1][:,
            None, :, :])
        rb = torch.min(rect1[:, :, 3][:, :, None, :], rect2[:, :, 3][:,
            None, :, :])
        wh = (rb - lt).clamp(min=0)
        non_rot_inter_areas = wh[:, :, :, 0] * wh[:, :, :, 1]
        non_rot_inter_areas = non_rot_inter_areas.view(B, K1, K2)
        if nums_k2 is not None:
            for b in range(B):
                non_rot_inter_areas[b, :, nums_k2[b]:] = 0
        enclosing_vols = enclosing_box3d_vol(corners1, corners2)
        vols1 = box3d_vol_tensor(corners1).clamp(min=EPS)
        vols2 = box3d_vol_tensor(corners2).clamp(min=EPS)
        sum_vols = vols1[:, :, None] + vols2[:, None, :]
        good_boxes = (enclosing_vols > 2 * EPS) * (sum_vols > 4 * EPS)
        if rotated_boxes:
            inter_areas = np.zeros((B, K1, K2), dtype=np.float32)
            rect1 = rect1.cpu().detach().numpy()
            rect2 = rect2.cpu().detach().numpy()
            nums_k2_np = nums_k2.cpu().numpy()
            non_rot_inter_areas_np = non_rot_inter_areas.cpu().detach().numpy()
            batch_intersect(rect1, rect2, non_rot_inter_areas_np,
                nums_k2_np, inter_areas, approximate)
            inter_areas = torch.from_numpy(inter_areas)
        else:
            inter_areas = non_rot_inter_areas
        inter_areas = inter_areas.to(corners1.device)
        inter_vols = inter_areas * height
        if return_inter_vols_only:
            return inter_vols
        union_vols = (sum_vols - inter_vols).clamp(min=EPS)
        ious = inter_vols / union_vols
        giou_second_term = -(1 - union_vols / enclosing_vols)
        gious = ious + giou_second_term
        gious *= good_boxes
        if nums_k2 is not None:
            mask = torch.zeros((B, K1, K2), device=height.device, dtype=
                torch.float32)
            for b in range(B):
                mask[b, :, :nums_k2[b]] = 1
            gious *= mask
        return gious
