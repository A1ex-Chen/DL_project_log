def __post_init__(self):
    super().__post_init__()
    if self.reference_image is None:
        raise ValueError(
            'Must provide a reference image when creating an Image2ImageRegion'
            )
    if self.strength < 0 or self.strength > 1:
        raise ValueError(
            f'The value of strength should in [0.0, 1.0] but is {self.strength}'
            )
    self.reference_image = resize(self.reference_image, size=[self.height,
        self.width])
