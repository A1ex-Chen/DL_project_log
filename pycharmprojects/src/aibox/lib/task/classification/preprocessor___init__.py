def __init__(self, image_resized_width: int, image_resized_height: int,
    image_min_side: int, image_max_side: int, image_side_divisor: int,
    eval_center_crop_ratio: float):
    super().__init__(image_resized_width, image_resized_height,
        image_min_side, image_max_side, image_side_divisor)
    self.eval_center_crop_ratio = eval_center_crop_ratio
