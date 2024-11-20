@torch.no_grad()
def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes,
    mask_gt):
    """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

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
        """
    self.bs = pd_scores.size(0)
    self.n_max_boxes = gt_bboxes.size(1)
    if self.n_max_boxes == 0:
        device = gt_bboxes.device
        return torch.full_like(pd_scores[..., 0], self.bg_idx).to(device
            ), torch.zeros_like(pd_bboxes).to(device), torch.zeros_like(
            pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(
            device)
    cycle, step, self.bs = (1, self.bs, self.bs
        ) if self.n_max_boxes <= 100 else (self.bs, 1, 1)
    target_labels_lst, target_bboxes_lst, target_scores_lst, fg_mask_lst = [
        ], [], [], []
    for i in range(cycle):
        start, end = i * step, (i + 1) * step
        pd_scores_ = pd_scores[start:end, ...]
        pd_bboxes_ = pd_bboxes[start:end, ...]
        gt_labels_ = gt_labels[start:end, ...]
        gt_bboxes_ = gt_bboxes[start:end, ...]
        mask_gt_ = mask_gt[start:end, ...]
        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores_,
            pd_bboxes_, gt_labels_, gt_bboxes_, anc_points, mask_gt_)
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos,
            overlaps, self.n_max_boxes)
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels_, gt_bboxes_, target_gt_idx, fg_mask)
        align_metric *= mask_pos
        pos_align_metrics = align_metric.max(axis=-1, keepdim=True)[0]
        pos_overlaps = (overlaps * mask_pos).max(axis=-1, keepdim=True)[0]
        norm_align_metric = (align_metric * pos_overlaps / (
            pos_align_metrics + self.eps)).max(-2)[0].unsqueeze(-1)
        target_scores = target_scores * norm_align_metric
        target_labels_lst.append(target_labels)
        target_bboxes_lst.append(target_bboxes)
        target_scores_lst.append(target_scores)
        fg_mask_lst.append(fg_mask)
    target_labels = torch.cat(target_labels_lst, 0)
    target_bboxes = torch.cat(target_bboxes_lst, 0)
    target_scores = torch.cat(target_scores_lst, 0)
    fg_mask = torch.cat(fg_mask_lst, 0)
    return target_labels, target_bboxes, target_scores, fg_mask.bool()
