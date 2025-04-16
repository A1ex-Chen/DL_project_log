def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints,
    batch_idx, stride_tensor, target_bboxes, pred_kpts):
    """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
    batch_idx = batch_idx.flatten()
    batch_size = len(masks)
    max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()
    batched_keypoints = torch.zeros((batch_size, max_kpts, keypoints.shape[
        1], keypoints.shape[2]), device=keypoints.device)
    for i in range(batch_size):
        keypoints_i = keypoints[batch_idx == i]
        batched_keypoints[i, :keypoints_i.shape[0]] = keypoints_i
    target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)
    selected_keypoints = batched_keypoints.gather(1, target_gt_idx_expanded
        .expand(-1, -1, keypoints.shape[1], keypoints.shape[2]))
    selected_keypoints /= stride_tensor.view(1, -1, 1, 1)
    kpts_loss = 0
    kpts_obj_loss = 0
    if masks.any():
        gt_kpt = selected_keypoints[masks]
        area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
        pred_kpt = pred_kpts[masks]
        kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1
            ] == 3 else torch.full_like(gt_kpt[..., 0], True)
        kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)
        if pred_kpt.shape[-1] == 3:
            kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())
    return kpts_loss, kpts_obj_loss
