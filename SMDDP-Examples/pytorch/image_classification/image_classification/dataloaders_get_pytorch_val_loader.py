def get_pytorch_val_loader(data_path, image_size, batch_size, num_classes,
    one_hot, interpolation='bilinear', workers=5, _worker_init_fn=None,
    crop_padding=32, memory_format=torch.contiguous_format):
    interpolation = {'bicubic': Image.BICUBIC, 'bilinear': Image.BILINEAR}[
        interpolation]
    valdir = os.path.join(data_path, 'val')
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(image_size + crop_padding, interpolation=
        interpolation), transforms.CenterCrop(image_size)]))
    if dist.is_initialized():
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, num_replicas=dist.get_world_size(),
            rank=dist.get_rank())
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_dataset, sampler=
        val_sampler, batch_size=batch_size, shuffle=val_sampler is None,
        num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=
        True, collate_fn=partial(fast_collate, memory_format), drop_last=
        False, persistent_workers=True)
    return PrefetchedWrapper(val_loader, 0, num_classes, one_hot), len(
        val_loader)
