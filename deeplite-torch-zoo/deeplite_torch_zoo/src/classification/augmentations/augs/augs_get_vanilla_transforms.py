def get_vanilla_transforms(img_size, hflip=0.5, jitter=0.4, mean=
    IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, crop_pct=
    DEFAULT_CROP_PCT, add_train_transforms=None, add_test_transforms=None,
    cutout_args=None, use_prefetcher=False):
    if isinstance(img_size, (tuple, list)):
        img_size = img_size[-1]
    train_transforms = [transforms.RandomResizedCrop(img_size), transforms.
        RandomHorizontalFlip(hflip), transforms.ColorJitter(brightness=
        jitter, contrast=jitter, saturation=jitter, hue=0)]
    if add_train_transforms is not None:
        train_transforms.append(add_train_transforms)
    if not use_prefetcher:
        train_transforms.append(transforms.ToTensor())
    else:
        train_transforms.append(ToNumpy())
    if cutout_args is not None:
        train_transforms.append(Cutout(**cutout_args))
    if not use_prefetcher:
        train_transforms.append(transforms.Normalize(mean, std))
    test_transforms = [transforms.Resize(int(img_size / crop_pct)),
        transforms.CenterCrop(img_size)]
    if add_test_transforms is not None:
        test_transforms.append(add_test_transforms)
    if not use_prefetcher:
        test_transforms += [transforms.ToTensor(), transforms.Normalize(
            mean, std)]
    else:
        test_transforms.append(ToNumpy())
    return transforms.Compose(train_transforms), transforms.Compose(
        test_transforms)
