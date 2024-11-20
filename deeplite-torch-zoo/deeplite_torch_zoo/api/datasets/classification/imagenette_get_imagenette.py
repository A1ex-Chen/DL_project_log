@DATASET_WRAPPER_REGISTRY.register(dataset_name='imagenette')
def get_imagenette(data_root=None, img_size=224, batch_size=64,
    test_batch_size=None, download=True, dataset_url=
    'https://github.com/ultralytics/yolov5/releases/download/v1.0/imagenette.zip'
    , use_prefetcher=False, num_workers=1, distributed=False, pin_memory=
    False, device=torch.device('cuda'), augmentation_mode='imagenet',
    train_transforms=None, val_transforms=None, mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD, crop_pct=DEFAULT_CROP_PCT, scale=(0.08, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0), hflip=0.5, vflip=0.0, color_jitter=0.4,
    auto_augment=None, train_interpolation='random', test_interpolation=
    'bilinear', re_prob=0.0, re_mode='pixel', re_count=1, re_split=False,
    num_aug_splits=0, num_aug_repeats=0, no_aug=False, collate_fn=None,
    use_multi_epochs_loader=False, worker_seeding='all'):
    return _get_imagenette(data_root=data_root, img_size=img_size,
        batch_size=batch_size, test_batch_size=test_batch_size, download=
        download, dataset_url=dataset_url, use_prefetcher=use_prefetcher,
        num_workers=num_workers, distributed=distributed, pin_memory=
        pin_memory, device=device, augmentation_mode=augmentation_mode,
        train_transforms=train_transforms, val_transforms=val_transforms,
        mean=mean, std=std, crop_pct=crop_pct, scale=scale, ratio=ratio,
        hflip=hflip, vflip=vflip, color_jitter=color_jitter, auto_augment=
        auto_augment, train_interpolation=train_interpolation,
        test_interpolation=test_interpolation, re_prob=re_prob, re_mode=
        re_mode, re_count=re_count, re_split=re_split, num_aug_splits=
        num_aug_splits, num_aug_repeats=num_aug_repeats, no_aug=no_aug,
        collate_fn=collate_fn, use_multi_epochs_loader=
        use_multi_epochs_loader, worker_seeding=worker_seeding)
