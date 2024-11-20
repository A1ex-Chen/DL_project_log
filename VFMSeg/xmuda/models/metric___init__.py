def __init__(self, num_classes, ignore_index=-100, name='seg_iou'):
    self.num_classes = num_classes
    self.ignore_index = ignore_index
    self.mat = None
    self.name = name
