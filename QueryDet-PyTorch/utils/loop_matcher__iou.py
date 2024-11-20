def _iou(self, boxes, box):
    iw = torch.clamp(boxes[:, 2], max=box[2]) - torch.clamp(boxes[:, 0],
        min=box[0])
    ih = torch.clamp(boxes[:, 3], max=box[3]) - torch.clamp(boxes[:, 1],
        min=box[1])
    inter = torch.clamp(iw, min=0) * torch.clamp(ih, min=0)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area = (box[2] - box[0]) * (box[3] - box[1])
    iou = inter / (areas + area - inter)
    return iou
