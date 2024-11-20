@property
def iou(self):
    h = self.mat.float()
    iou = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
    return iou
