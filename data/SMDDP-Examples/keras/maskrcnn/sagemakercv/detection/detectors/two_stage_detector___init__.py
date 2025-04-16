def __init__(self, backbone, neck, dense_head, roi_head,
    global_gradient_clip_ratio=0.0, weight_decay=0.0):
    super(TwoStageDetector, self).__init__()
    self.backbone = backbone
    self.neck = neck
    self.rpn_head = dense_head
    self.roi_head = roi_head
    self.global_gradient_clip_ratio = global_gradient_clip_ratio
    self.weight_decay = weight_decay
