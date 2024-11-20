def classify_augmentations(size=224, mean=DEFAULT_MEAN, std=DEFAULT_STD,
    scale=None, ratio=None, hflip=0.5, vflip=0.0, auto_augment=None, hsv_h=
    0.015, hsv_s=0.4, hsv_v=0.4, force_color_jitter=False, erasing=0.0,
    interpolation=Image.BILINEAR):
    """
    Classification transforms with augmentation for training. Inspired by timm/data/transforms_factory.py.

    Args:
        size (int): image size
        scale (tuple): scale range of the image. default is (0.08, 1.0)
        ratio (tuple): aspect ratio range of the image. default is (3./4., 4./3.)
        mean (tuple): mean values of RGB channels
        std (tuple): std values of RGB channels
        hflip (float): probability of horizontal flip
        vflip (float): probability of vertical flip
        auto_augment (str): auto augmentation policy. can be 'randaugment', 'augmix', 'autoaugment' or None.
        hsv_h (float): image HSV-Hue augmentation (fraction)
        hsv_s (float): image HSV-Saturation augmentation (fraction)
        hsv_v (float): image HSV-Value augmentation (fraction)
        force_color_jitter (bool): force to apply color jitter even if auto augment is enabled
        erasing (float): probability of random erasing
        interpolation (T.InterpolationMode): interpolation mode. default is T.InterpolationMode.BILINEAR.

    Returns:
        (T.Compose): torchvision transforms
    """
    import torchvision.transforms as T
    if not isinstance(size, int):
        raise TypeError(
            f'classify_transforms() size {size} must be integer, not (list, tuple)'
            )
    scale = tuple(scale or (0.08, 1.0))
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))
    primary_tfl = [T.RandomResizedCrop(size, scale=scale, ratio=ratio,
        interpolation=interpolation)]
    if hflip > 0.0:
        primary_tfl.append(T.RandomHorizontalFlip(p=hflip))
    if vflip > 0.0:
        primary_tfl.append(T.RandomVerticalFlip(p=vflip))
    secondary_tfl = []
    disable_color_jitter = False
    if auto_augment:
        assert isinstance(auto_augment, str
            ), f'Provided argument should be string, but got type {type(auto_augment)}'
        disable_color_jitter = not force_color_jitter
        if auto_augment == 'randaugment':
            if TORCHVISION_0_11:
                secondary_tfl.append(T.RandAugment(interpolation=interpolation)
                    )
            else:
                LOGGER.warning(
                    '"auto_augment=randaugment" requires torchvision >= 0.11.0. Disabling it.'
                    )
        elif auto_augment == 'augmix':
            if TORCHVISION_0_13:
                secondary_tfl.append(T.AugMix(interpolation=interpolation))
            else:
                LOGGER.warning(
                    '"auto_augment=augmix" requires torchvision >= 0.13.0. Disabling it.'
                    )
        elif auto_augment == 'autoaugment':
            if TORCHVISION_0_10:
                secondary_tfl.append(T.AutoAugment(interpolation=interpolation)
                    )
            else:
                LOGGER.warning(
                    '"auto_augment=autoaugment" requires torchvision >= 0.10.0. Disabling it.'
                    )
        else:
            raise ValueError(
                f'Invalid auto_augment policy: {auto_augment}. Should be one of "randaugment", "augmix", "autoaugment" or None'
                )
    if not disable_color_jitter:
        secondary_tfl.append(T.ColorJitter(brightness=hsv_v, contrast=hsv_v,
            saturation=hsv_s, hue=hsv_h))
    final_tfl = [T.ToTensor(), T.Normalize(mean=torch.tensor(mean), std=
        torch.tensor(std)), T.RandomErasing(p=erasing, inplace=True)]
    return T.Compose(primary_tfl + secondary_tfl + final_tfl)
