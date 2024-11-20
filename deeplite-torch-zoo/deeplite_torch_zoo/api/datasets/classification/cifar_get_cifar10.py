@DATASET_WRAPPER_REGISTRY.register(dataset_name='cifar10')
def get_cifar10(data_root=None, img_size=CIFAR_IMAGE_SIZE, batch_size=64,
    test_batch_size=None, download=True, use_prefetcher=False, num_workers=
    1, distributed=False, pin_memory=False, device=torch.device('cuda'),
    train_transforms=None, val_transforms=None, mean=CIFAR_MEAN, std=
    CIFAR_STD, padding=4, re_prob=0.0, re_mode='pixel', re_count=1,
    num_aug_repeats=0, no_aug=False, collate_fn=None,
    use_multi_epochs_loader=False, worker_seeding='all'):
    cifar_cls = torchvision.datasets.CIFAR10
    return _get_cifar(cifar_cls, data_root, img_size=img_size, batch_size=
        batch_size, test_batch_size=test_batch_size, download=download,
        use_prefetcher=use_prefetcher, num_workers=num_workers, distributed
        =distributed, pin_memory=pin_memory, device=device,
        train_transforms=train_transforms, val_transforms=val_transforms,
        mean=mean, std=std, padding=padding, re_prob=re_prob, re_mode=
        re_mode, re_count=re_count, num_aug_repeats=num_aug_repeats, no_aug
        =no_aug, collate_fn=collate_fn, use_multi_epochs_loader=
        use_multi_epochs_loader, worker_seeding=worker_seeding)
