def get_imagenet_transforms(img_size, mean=IMAGENET_DEFAULT_MEAN, std=
    IMAGENET_DEFAULT_STD, crop_pct=DEFAULT_CROP_PCT, scale=(0.08, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0), hflip=0.5, vflip=0.0, color_jitter=0.4,
    auto_augment=None, train_interpolation='random', test_interpolation=
    'bilinear', re_prob=0.0, re_mode='pixel', re_count=1, re_num_splits=0,
    use_prefetcher=False):
    if isinstance(img_size, (tuple, list)):
        img_size = img_size[-1]
    train_transforms = transforms_imagenet_train(img_size, mean=mean, std=
        std, scale=scale, ratio=ratio, hflip=hflip, vflip=vflip,
        color_jitter=color_jitter, auto_augment=auto_augment, interpolation
        =train_interpolation, re_prob=re_prob, re_mode=re_mode, re_count=
        re_count, re_num_splits=re_num_splits, use_prefetcher=use_prefetcher)
    val_transforms = transforms_imagenet_eval(img_size, mean=mean, std=std,
        crop_pct=crop_pct, interpolation=test_interpolation, use_prefetcher
        =use_prefetcher)
    return train_transforms, val_transforms
