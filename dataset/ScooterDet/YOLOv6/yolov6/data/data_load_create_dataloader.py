def create_dataloader(path, img_size, batch_size, stride, hyp=None, augment
    =False, check_images=False, check_labels=False, pad=0.0, rect=False,
    rank=-1, workers=8, shuffle=False, data_dict=None, task='Train',
    specific_shape=False, height=1088, width=1920, cache_ram=False):
    """Create general dataloader.

    Returns dataloader and dataset
    """
    if rect and shuffle:
        LOGGER.warning(
            'WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False'
            )
        shuffle = False
    with torch_distributed_zero_first(rank):
        dataset = TrainValDataset(path, img_size, batch_size, augment=
            augment, hyp=hyp, rect=rect, check_images=check_images,
            check_labels=check_labels, stride=int(stride), pad=pad, rank=
            rank, data_dict=data_dict, task=task, specific_shape=
            specific_shape, height=height, width=width, cache_ram=cache_ram)
    batch_size = min(batch_size, len(dataset))
    workers = min([os.cpu_count() // int(os.getenv('WORLD_SIZE', 1)), 
        batch_size if batch_size > 1 else 0, workers])
    drop_last = rect and dist.is_initialized() and dist.get_world_size() > 1
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset,
        shuffle=shuffle, drop_last=drop_last)
    return TrainValDataLoader(dataset, batch_size=batch_size, shuffle=
        shuffle and sampler is None, num_workers=workers, sampler=sampler,
        pin_memory=True, collate_fn=TrainValDataset.collate_fn), dataset
