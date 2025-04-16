def classify_transforms(size=224, mean=DEFAULT_MEAN, std=DEFAULT_STD,
    interpolation=Image.BILINEAR, crop_fraction: float=DEFAULT_CROP_FRACTION):
    """
    Classification transforms for evaluation/inference. Inspired by timm/data/transforms_factory.py.

    Args:
        size (int): image size
        mean (tuple): mean values of RGB channels
        std (tuple): std values of RGB channels
        interpolation (T.InterpolationMode): interpolation mode. default is T.InterpolationMode.BILINEAR.
        crop_fraction (float): fraction of image to crop. default is 1.0.

    Returns:
        (T.Compose): torchvision transforms
    """
    import torchvision.transforms as T
    if isinstance(size, (tuple, list)):
        assert len(size
            ) == 2, f"'size' tuples must be length 2, not length {len(size)}"
        scale_size = tuple(math.floor(x / crop_fraction) for x in size)
    else:
        scale_size = math.floor(size / crop_fraction)
        scale_size = scale_size, scale_size
    if scale_size[0] == scale_size[1]:
        tfl = [T.Resize(scale_size[0], interpolation=interpolation)]
    else:
        tfl = [T.Resize(scale_size)]
    tfl.extend([T.CenterCrop(size), T.ToTensor(), T.Normalize(mean=torch.
        tensor(mean), std=torch.tensor(std))])
    return T.Compose(tfl)
