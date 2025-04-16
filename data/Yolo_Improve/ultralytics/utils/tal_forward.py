@torch.no_grad()
def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes,
    mask_gt):
    """
        Compute the task-aligned assignment. Reference code is available at
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
    self.bs = pd_scores.shape[0]
    self.n_max_boxes = gt_bboxes.shape[1]
    if self.n_max_boxes == 0:
        device = gt_bboxes.device
        return torch.full_like(pd_scores[..., 0], self.bg_idx).to(device
            ), torch.zeros_like(pd_bboxes).to(device), torch.zeros_like(
            pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(
            device), torch.zeros_like(pd_scores[..., 0]).to(device)
    mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores,
        pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)
    target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos,
        overlaps, self.n_max_boxes)
    target_labels, target_bboxes, target_scores = self.get_targets(gt_labels,
        gt_bboxes, target_gt_idx, fg_mask)
    align_metric *= mask_pos
    pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
    pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
    norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics +
        self.eps)).amax(-2).unsqueeze(-1)
    target_scores = target_scores * norm_align_metric
    return target_labels, target_bboxes, target_scores, fg_mask.bool(
        ), target_gt_idx
