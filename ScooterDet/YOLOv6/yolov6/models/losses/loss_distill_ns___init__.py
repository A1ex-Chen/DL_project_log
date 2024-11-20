def __init__(self, num_classes, reg_max, use_dfl=False, iou_type='giou'):
    super(BboxLoss, self).__init__()
    self.num_classes = num_classes
    self.iou_loss = IOUloss(box_format='xyxy', iou_type=iou_type, eps=1e-10)
    self.reg_max = reg_max
    self.use_dfl = use_dfl
