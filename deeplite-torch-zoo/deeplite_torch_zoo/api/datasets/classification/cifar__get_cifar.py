def _get_cifar(cifar_cls, data_root=None, img_size=CIFAR_IMAGE_SIZE,
    batch_size=64, test_batch_size=None, download=True, use_prefetcher=
    False, num_workers=1, distributed=False, pin_memory=False, device=torch
    .device('cuda'), train_transforms=None, val_transforms=None, mean=
    CIFAR_MEAN, std=CIFAR_STD, padding=4, re_prob=0.0, re_mode='pixel',
    re_count=1, num_aug_repeats=0, no_aug=False, collate_fn=None,
    use_multi_epochs_loader=False, worker_seeding='all'):
    if isinstance(device, str):
        device = torch.device(device)
    if data_root is None:
        data_root = os.path.join(expanduser('~'), '.deeplite-torch-zoo')
    default_train_transforms = transforms.Compose([transforms.RandomCrop(
        img_size, padding=padding), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    default_val_transforms = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    dataset_train = cifar_cls(root=data_root, train=True, download=download,
        transform=train_transforms or default_train_transforms)
    dataset_eval = cifar_cls(root=data_root, train=False, download=download,
        transform=val_transforms or default_val_transforms)
    train_loader = create_loader(dataset_train, input_size=img_size,
        batch_size=batch_size, is_training=True, use_prefetcher=
        use_prefetcher, no_aug=no_aug, re_prob=re_prob, re_mode=re_mode,
        re_count=re_count, num_aug_repeats=num_aug_repeats, mean=mean, std=
        std, num_workers=num_workers, distributed=distributed, collate_fn=
        collate_fn, pin_memory=pin_memory, device=device,
        use_multi_epochs_loader=use_multi_epochs_loader, worker_seeding=
        worker_seeding)
    test_loader = create_loader(dataset_eval, input_size=img_size,
        batch_size=test_batch_size or batch_size, is_training=False,
        use_prefetcher=use_prefetcher, mean=mean, std=std, num_workers=
        num_workers, distributed=distributed, pin_memory=pin_memory, device
        =device)
    return {'train': train_loader, 'test': test_loader}
