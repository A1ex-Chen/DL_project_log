def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes,
    anc_points, mask_gt):
    """Get in_gts mask, (b, max_num_obj, h*w)."""
    mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
    align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes,
        gt_labels, gt_bboxes, mask_in_gts * mask_gt)
    mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt
        .expand(-1, -1, self.topk).bool())
    mask_pos = mask_topk * mask_in_gts * mask_gt
    return mask_pos, align_metric, overlaps
