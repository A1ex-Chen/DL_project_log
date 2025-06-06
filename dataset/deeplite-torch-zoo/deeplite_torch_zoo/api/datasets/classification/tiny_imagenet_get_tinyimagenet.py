@DATASET_WRAPPER_REGISTRY.register(dataset_name='tinyimagenet')
def get_tinyimagenet(data_root=None, img_size=64, batch_size=64,
    test_batch_size=None, use_prefetcher=False, num_workers=1, distributed=
    False, pin_memory=False, device=torch.device('cuda'), augmentation_mode
    ='vanilla', train_transforms=None, val_transforms=None, mean=
    TINYIMAGENET_MEAN, std=TINYIMAGENET_STD, crop_pct=1.0, scale=(0.08, 1.0
    ), ratio=(3.0 / 4.0, 4.0 / 3.0), hflip=0.5, vflip=0.0, color_jitter=0.4,
    auto_augment=None, train_interpolation='random', test_interpolation=
    'bilinear', re_prob=0.0, re_mode='pixel', re_count=1, re_split=False,
    num_aug_splits=0, num_aug_repeats=0, no_aug=False, collate_fn=None,
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
    data_transforms = {'train': train_transforms or
        default_train_transforms, 'val': val_transforms or
        default_val_transforms}
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_root, x),
        data_transforms[x]) for x in ['train', 'val']}
    train_loader = create_loader(image_datasets['train'], input_size=
        img_size, batch_size=batch_size, is_training=True, use_prefetcher=
        use_prefetcher, no_aug=no_aug, re_prob=re_prob, re_mode=re_mode,
        re_count=re_count, num_aug_repeats=num_aug_repeats, re_num_splits=
        re_num_splits, mean=mean, std=std, num_workers=num_workers,
        distributed=distributed, collate_fn=collate_fn, pin_memory=
        pin_memory, device=device, use_multi_epochs_loader=
        use_multi_epochs_loader, worker_seeding=worker_seeding)
    test_loader = create_loader(image_datasets['val'], input_size=img_size,
        batch_size=test_batch_size or batch_size, is_training=False,
        use_prefetcher=use_prefetcher, mean=mean, std=std, num_workers=
        num_workers, distributed=distributed, pin_memory=pin_memory, device
        =device)
    return {'train': train_loader, 'test': test_loader}
