def get_pytorch_train_loader(data_path, image_size, batch_size, num_classes,
    one_hot, interpolation='bilinear', augmentation=None, start_epoch=0,
    workers=5, _worker_init_fn=None, memory_format=torch.contiguous_format):
    interpolation = {'bicubic': Image.BICUBIC, 'bilinear': Image.BILINEAR}[
        interpolation]
    traindir = os.path.join(data_path, 'train')
    transforms_list = [transforms.RandomResizedCrop(image_size,
        interpolation=interpolation), transforms.RandomHorizontalFlip()]
    if augmentation == 'autoaugment':
        transforms_list.append(AutoaugmentImageNetPolicy())
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose(
        transforms_list))
    if dist.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, num_replicas=dist.get_world_size(),
            rank=dist.get_rank())
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=
        train_sampler, batch_size=batch_size, shuffle=train_sampler is None,
        num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=
        True, collate_fn=partial(fast_collate, memory_format), drop_last=
        True, persistent_workers=True)
    return PrefetchedWrapper(train_loader, start_epoch, num_classes, one_hot
        ), len(train_loader)
