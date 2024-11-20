def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
    """Compute alignment metric given predicted and ground truth bounding boxes."""
    na = pd_bboxes.shape[-2]
    mask_gt = mask_gt.bool()
    overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes
        .dtype, device=pd_bboxes.device)
    bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=
        pd_scores.dtype, device=pd_scores.device)
    ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
    ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
    ind[1] = gt_labels.squeeze(-1)
    bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]
    pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[
        mask_gt]
    gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
    overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)
    align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
    return align_metric, overlaps
