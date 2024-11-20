def __init__(self, topk=9, num_classes=80):
    super(ATSSAssigner, self).__init__()
    self.topk = topk
    self.num_classes = num_classes
    self.bg_idx = num_classes
