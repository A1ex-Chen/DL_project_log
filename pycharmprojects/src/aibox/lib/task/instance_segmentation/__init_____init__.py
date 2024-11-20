def __init__(self, num_classes: int, image_min_side: int, image_max_side: int):
    super().__init__()
    self.num_classes = num_classes
    self.image_min_side = image_min_side
    self.image_max_side = image_max_side
