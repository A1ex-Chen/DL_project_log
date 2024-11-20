def is_valid_image(image):
    return isinstance(image, PIL.Image.Image) or isinstance(image, (np.
        ndarray, torch.Tensor)) and image.ndim in (2, 3)
