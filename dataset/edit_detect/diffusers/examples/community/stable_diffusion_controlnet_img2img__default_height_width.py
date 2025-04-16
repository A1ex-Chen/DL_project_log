def _default_height_width(self, height, width, image):
    if isinstance(image, list):
        image = image[0]
    if height is None:
        if isinstance(image, PIL.Image.Image):
            height = image.height
        elif isinstance(image, torch.Tensor):
            height = image.shape[3]
        height = height // 8 * 8
    if width is None:
        if isinstance(image, PIL.Image.Image):
            width = image.width
        elif isinstance(image, torch.Tensor):
            width = image.shape[2]
        width = width // 8 * 8
    return height, width
