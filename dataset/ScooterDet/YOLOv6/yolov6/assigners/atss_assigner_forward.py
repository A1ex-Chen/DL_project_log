@torch.no_grad()
def forward(self, anc_bboxes, n_level_bboxes, gt_labels, gt_bboxes, mask_gt,
    pd_bboxes):
    """This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

        Args:
            anc_bboxes (Tensor): shape(num_total_anchors, 4)
            n_level_bboxes (List):len(3)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
            pd_bboxes (Tensor): shape(bs, n_max_boxes, 4)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
    self.n_anchors = anc_bboxes.size(0)
    self.bs = gt_bboxes.size(0)
    self.n_max_boxes = gt_bboxes.size(1)
    if self.n_max_boxes == 0:
        device = gt_bboxes.device
        return torch.full([self.bs, self.n_anchors], self.bg_idx).to(device
            ), torch.zeros([self.bs, self.n_anchors, 4]).to(device
            ), torch.zeros([self.bs, self.n_anchors, self.num_classes]).to(
            device), torch.zeros([self.bs, self.n_anchors]).to(device)
    overlaps = iou2d_calculator(gt_bboxes.reshape([-1, 4]), anc_bboxes)
    overlaps = overlaps.reshape([self.bs, -1, self.n_anchors])
    distances, ac_points = dist_calculator(gt_bboxes.reshape([-1, 4]),
        anc_bboxes)
    distances = distances.reshape([self.bs, -1, self.n_anchors])
    is_in_candidate, candidate_idxs = self.select_topk_candidates(distances,
        n_level_bboxes, mask_gt)
    overlaps_thr_per_gt, iou_candidates = self.thres_calculator(is_in_candidate
        , candidate_idxs, overlaps)
    is_pos = torch.where(iou_candidates > overlaps_thr_per_gt.repeat([1, 1,
        self.n_anchors]), is_in_candidate, torch.zeros_like(is_in_candidate))
    is_in_gts = select_candidates_in_gts(ac_points, gt_bboxes)
    mask_pos = is_pos * is_in_gts * mask_gt
    target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos,
        overlaps, self.n_max_boxes)
    target_labels, target_bboxes, target_scores = self.get_targets(gt_labels,
        gt_bboxes, target_gt_idx, fg_mask)
    if pd_bboxes is not None:
        ious = iou_calculator(gt_bboxes, pd_bboxes) * mask_pos
        ious = ious.max(axis=-2)[0].unsqueeze(-1)
        target_scores *= ious
    return target_labels.long(), target_bboxes, target_scores, fg_mask.bool()
