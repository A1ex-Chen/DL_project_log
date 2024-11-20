def iou_calculation(self, gt_bboxes, pd_bboxes):
    """IoU calculation for rotated bounding boxes."""
    return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)
