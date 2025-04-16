def _get_assigned_bboxes(self, pred_bboxes, gt_bboxes, match_indices):
    """Assigns predicted bounding boxes to ground truth bounding boxes based on the match indices."""
    pred_assigned = torch.cat([(t[i] if len(i) > 0 else torch.zeros(0, t.
        shape[-1], device=self.device)) for t, (i, _) in zip(pred_bboxes,
        match_indices)])
    gt_assigned = torch.cat([(t[j] if len(j) > 0 else torch.zeros(0, t.
        shape[-1], device=self.device)) for t, (_, j) in zip(gt_bboxes,
        match_indices)])
    return pred_assigned, gt_assigned
