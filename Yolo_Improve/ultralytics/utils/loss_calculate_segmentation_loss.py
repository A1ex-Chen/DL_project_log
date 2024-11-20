def calculate_segmentation_loss(self, fg_mask: torch.Tensor, masks: torch.
    Tensor, target_gt_idx: torch.Tensor, target_bboxes: torch.Tensor,
    batch_idx: torch.Tensor, proto: torch.Tensor, pred_masks: torch.Tensor,
    imgsz: torch.Tensor, overlap: bool) ->torch.Tensor:
    """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
    _, _, mask_h, mask_w = proto.shape
    loss = 0
    target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
    marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)
    mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w,
        mask_h], device=proto.device)
    for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks,
        proto, mxyxy, marea, masks)):
        (fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i,
            marea_i, masks_i) = single_i
        if fg_mask_i.any():
            mask_idx = target_gt_idx_i[fg_mask_i]
            if overlap:
                gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                gt_mask = gt_mask.float()
            else:
                gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
            loss += self.single_mask_loss(gt_mask, pred_masks_i[fg_mask_i],
                proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i])
        else:
            loss += (proto * 0).sum() + (pred_masks * 0).sum()
    return loss / fg_mask.sum()
