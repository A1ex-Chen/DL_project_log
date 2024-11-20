def iou_score(bbox_det: list[int], bbox_gt: list[int]) ->float:
    x1_det, y1_det, x2_det, y2_det = bbox_det
    x1_gt, y1_gt, x2_gt, y2_gt = bbox_gt
    inter_area_x = max(min(x2_det, x2_gt) - max(x1_det, x1_gt), 0)
    inter_area_y = max(min(y2_det, y2_gt) - max(y1_det, y1_gt), 0)
    intersection = inter_area_x * inter_area_y
    union = (x2_det - x1_det) * (y2_det - y1_det) + (x2_gt - x1_gt) * (y2_gt -
        y1_gt) - intersection
    iou = intersection / union
    return iou
