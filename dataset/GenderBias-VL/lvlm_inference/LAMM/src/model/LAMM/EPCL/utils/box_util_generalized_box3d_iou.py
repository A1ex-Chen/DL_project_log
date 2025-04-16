def generalized_box3d_iou(corners1: torch.Tensor, corners2: torch.Tensor,
    nums_k2: torch.Tensor, rotated_boxes: bool=True, return_inter_vols_only:
    bool=False, needs_grad: bool=False):
    if needs_grad is True or box_intersection is None:
        context = torch.enable_grad if needs_grad else torch.no_grad
        with context():
            return generalized_box3d_iou_tensor_jit(corners1, corners2,
                nums_k2, rotated_boxes, return_inter_vols_only)
    else:
        with torch.no_grad():
            return generalized_box3d_iou_cython(corners1, corners2, nums_k2,
                rotated_boxes, return_inter_vols_only)
