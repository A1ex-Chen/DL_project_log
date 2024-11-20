def create_dataloader(path, imgsz, batch_size, stride, single_cls=False,
    hyp=None, augment=False, cache=False, pad=0.0, rect=False, rank=-1,
    workers=8, image_weights=False, quad=False, prefix='', shuffle=False,
    seed=0):
    if rect and shuffle:
        LOGGER.warning(
            'WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False'
            )
        shuffle = False
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size, augment=
            augment, hyp=hyp, rect=rect, cache_images=cache, single_cls=
            single_cls, stride=int(stride), pad=pad, image_weights=
            image_weights, prefix=prefix)
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else
        0, workers])
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset,
        shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    return loader(dataset, batch_size=batch_size, shuffle=shuffle and 
        sampler is None, num_workers=nw, sampler=sampler, pin_memory=
        PIN_MEMORY, collate_fn=LoadImagesAndLabels.collate_fn4 if quad else
        LoadImagesAndLabels.collate_fn, worker_init_fn=seed_worker,
        generator=generator), dataset
