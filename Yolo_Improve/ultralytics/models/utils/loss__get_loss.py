def _get_loss(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
    masks=None, gt_mask=None, postfix='', match_indices=None):
    """Get losses."""
    if match_indices is None:
        match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes,
            gt_cls, gt_groups, masks=masks, gt_mask=gt_mask)
    idx, gt_idx = self._get_index(match_indices)
    pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]
    bs, nq = pred_scores.shape[:2]
    targets = torch.full((bs, nq), self.nc, device=pred_scores.device,
        dtype=gt_cls.dtype)
    targets[idx] = gt_cls[gt_idx]
    gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
    if len(gt_bboxes):
        gt_scores[idx] = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True
            ).squeeze(-1)
    loss = {}
    loss.update(self._get_loss_class(pred_scores, targets, gt_scores, len(
        gt_bboxes), postfix))
    loss.update(self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix))
    return loss
