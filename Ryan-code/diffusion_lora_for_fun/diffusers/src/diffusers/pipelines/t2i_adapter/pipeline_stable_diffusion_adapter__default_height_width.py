def _default_height_width(self, height, width, image):
    while isinstance(image, list):
        image = image[0]
    if height is None:
        if isinstance(image, PIL.Image.Image):
            height = image.height
        elif isinstance(image, torch.Tensor):
            height = image.shape[-2]
        height = (height // self.adapter.downscale_factor * self.adapter.
            downscale_factor)
    if width is None:
        if isinstance(image, PIL.Image.Image):
            width = image.width
        elif isinstance(image, torch.Tensor):
            width = image.shape[-1]
        width = (width // self.adapter.downscale_factor * self.adapter.
            downscale_factor)
    return height, width
