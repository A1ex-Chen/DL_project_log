def __str__(self):
    return '{iou:.4f}'.format(iou=self.iou.mean().item())
