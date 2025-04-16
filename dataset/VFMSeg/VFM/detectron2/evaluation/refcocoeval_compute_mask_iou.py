def compute_mask_iou(outputs: torch.Tensor, labels: torch.Tensor, EPS=1e-06):
    outputs = outputs.int()
    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))
    iou = (intersection + EPS) / (union + EPS)
    return iou, intersection, union
