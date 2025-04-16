def __init__(self, image_resized_width: int, image_resized_height: int,
    image_min_side: int, image_max_side: int, image_side_divisor: int):
    super().__init__()
    self.image_resized_width = image_resized_width
    self.image_resized_height = image_resized_height
    self.image_min_side = image_min_side
    self.image_max_side = image_max_side
    self.image_side_divisor = image_side_divisor
