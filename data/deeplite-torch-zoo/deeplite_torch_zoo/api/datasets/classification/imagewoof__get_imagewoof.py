def _get_imagewoof(data_root=None, img_size=224, batch_size=64,
    test_batch_size=None, download=True, dataset_url=None, use_prefetcher=
    False, map_to_imagenet_labels=False, num_workers=1, distributed=False,
    pin_memory=False, device=torch.device('cuda'), augmentation_mode=
    'imagenet', train_transforms=None, val_transforms=None, mean=
    IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, crop_pct=
    DEFAULT_CROP_PCT, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
    hflip=0.5, vflip=0.0, color_jitter=0.4, auto_augment=None,
    train_interpolation='random', test_interpolation='bilinear', re_prob=
    0.0, re_mode='pixel', re_count=1, re_split=False, num_aug_splits=0,
    num_aug_repeats=0, no_aug=False, collate_fn=None,
    use_multi_epochs_loader=False, worker_seeding='all'):
    if isinstance(device, str):
        device = torch.device(device)
    if data_root is None:
        data_root = os.path.join(expanduser('~'), '.deeplite-torch-zoo')
    if augmentation_mode not in ('vanilla', 'imagenet'):
        raise ValueError(
            f'Wrong value of augmentation_mode arg: {augmentation_mode}. Choices: "vanilla", "imagenet"'
            )
    re_num_splits = 0
    if re_split:
        re_num_splits = num_aug_splits or 2
    if augmentation_mode == 'imagenet':
        default_train_transforms, default_val_transforms = (
            get_imagenet_transforms(img_size, mean=mean, std=std, crop_pct=
            crop_pct, scale=scale, ratio=ratio, hflip=hflip, vflip=vflip,
            color_jitter=color_jitter, auto_augment=auto_augment,
            train_interpolation=train_interpolation, test_interpolation=
            test_interpolation, re_prob=re_prob, re_mode=re_mode, re_count=
            re_count, re_num_splits=re_num_splits, use_prefetcher=
            use_prefetcher))
    else:
        default_train_transforms, default_val_transforms = (
            get_vanilla_transforms(img_size, hflip=hflip, jitter=
            color_jitter, mean=mean, std=std, crop_pct=crop_pct,
            use_prefetcher=use_prefetcher))
    dataset_train = Imagewoof(root=data_root, split='train', download=
        download, transform=train_transforms or default_train_transforms,
        url=dataset_url, map_to_imagenet_labels=map_to_imagenet_labels)
    dataset_eval = Imagewoof(root=data_root, split='val', download=download,
        transform=val_transforms or default_val_transforms, url=dataset_url,
        map_to_imagenet_labels=map_to_imagenet_labels)
    train_loader = create_loader(dataset_train, input_size=img_size,
        batch_size=batch_size, is_training=True, use_prefetcher=
        use_prefetcher, no_aug=no_aug, re_prob=re_prob, re_mode=re_mode,
        re_count=re_count, num_aug_repeats=num_aug_repeats, re_num_splits=
        re_num_splits, mean=mean, std=std, num_workers=num_workers,
        distributed=distributed, collate_fn=collate_fn, pin_memory=
        pin_memory, device=device, use_multi_epochs_loader=
        use_multi_epochs_loader, worker_seeding=worker_seeding)
    test_loader = create_loader(dataset_eval, input_size=img_size,
        batch_size=test_batch_size or batch_size, is_training=False,
        use_prefetcher=use_prefetcher, mean=mean, std=std, num_workers=
        num_workers, distributed=distributed, pin_memory=pin_memory, device
        =device)
    return {'train': train_loader, 'test': test_loader}
