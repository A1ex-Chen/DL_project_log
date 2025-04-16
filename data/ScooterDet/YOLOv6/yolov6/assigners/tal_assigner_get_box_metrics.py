def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
    pd_scores = pd_scores.permute(0, 2, 1)
    gt_labels = gt_labels.to(torch.long)
    ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
    ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)
    ind[1] = gt_labels.squeeze(-1)
    bbox_scores = pd_scores[ind[0], ind[1]]
    overlaps = iou_calculator(gt_bboxes, pd_bboxes)
    align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
    return align_metric, overlaps
