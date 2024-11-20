def __init__(self, anchor_num, num_classes, score_th=0.12, context=2):
    self.anchor_num = anchor_num
    self.num_classes = num_classes
    self.score_th = score_th
    self.context = context
    self.initialized = False
    self.cls_spconv = None
    self.bbox_spconv = None
    self.qcls_spconv = None
    self.qcls_conv = None
    self.n_conv = None
