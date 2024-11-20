def get_iou(gt_masks, pred_masks, ignore_label=-1):
    rev_ignore_mask = ~(gt_masks == ignore_label)
    gt_masks = gt_masks.bool()
    n, h, w = gt_masks.shape
    intersection = (gt_masks & pred_masks & rev_ignore_mask).reshape(n, h * w
        ).sum(dim=-1)
    union = ((gt_masks | pred_masks) & rev_ignore_mask).reshape(n, h * w).sum(
        dim=-1)
    ious = intersection / union
    return ious
