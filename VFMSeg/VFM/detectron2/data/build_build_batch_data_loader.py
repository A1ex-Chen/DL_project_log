def build_batch_data_loader(dataset, sampler, total_batch_size, *,
    aspect_ratio_grouping=False, num_workers=0, collate_fn=None):
    """
    Build a batched dataloader. The main differences from `torch.utils.data.DataLoader` are:
    1. support aspect ratio grouping options
    2. use no "batch collation", because this is common for detection training

    Args:
        dataset (torch.utils.data.Dataset): a pytorch map-style or iterable dataset.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces indices.
            Must be provided iff. ``dataset`` is a map-style dataset.
        total_batch_size, aspect_ratio_grouping, num_workers, collate_fn: see
            :func:`build_detection_train_loader`.

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    assert total_batch_size > 0 and total_batch_size % world_size == 0, 'Total batch size ({}) must be divisible by the number of gpus ({}).'.format(
        total_batch_size, world_size)
    batch_size = total_batch_size // world_size
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, 'sampler must be None if dataset is IterableDataset'
    else:
        dataset = ToIterableDataset(dataset, sampler)
    if aspect_ratio_grouping:
        data_loader = torchdata.DataLoader(dataset, num_workers=num_workers,
            collate_fn=operator.itemgetter(0), worker_init_fn=
            worker_init_reset_seed)
        data_loader = AspectRatioGroupedDataset(data_loader, batch_size)
        if collate_fn is None:
            return data_loader
        return MapDataset(data_loader, collate_fn)
    else:
        return torchdata.DataLoader(dataset, batch_size=batch_size,
            drop_last=True, num_workers=num_workers, collate_fn=
            trivial_batch_collator if collate_fn is None else collate_fn,
            worker_init_fn=worker_init_reset_seed)
