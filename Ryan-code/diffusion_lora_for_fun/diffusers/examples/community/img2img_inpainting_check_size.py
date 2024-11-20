def check_size(image, height, width):
    if isinstance(image, PIL.Image.Image):
        w, h = image.size
    elif isinstance(image, torch.Tensor):
        *_, h, w = image.shape
    if h != height or w != width:
        raise ValueError(
            f'Image size should be {height}x{width}, but got {h}x{w}')
