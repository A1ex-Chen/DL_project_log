def __init__(self, pretrained: bool, num_frozen_levels: int):
    super().__init__()
    self.pretrained = pretrained
    self.num_frozen_levels = num_frozen_levels
    self.component = self._build_component()
    self._freeze_layers()
