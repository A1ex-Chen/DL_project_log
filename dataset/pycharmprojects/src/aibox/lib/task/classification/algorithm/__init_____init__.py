def __init__(self, num_classes: int, pretrained: bool, num_frozen_levels:
    int, eval_center_crop_ratio: float):
    super().__init__()
    self.num_classes = num_classes
    self.pretrained = pretrained
    self.num_frozen_levels = num_frozen_levels
    self.eval_center_crop_ratio = eval_center_crop_ratio
    self.net = self._build_net()
