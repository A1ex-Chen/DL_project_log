def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None,
    augment=False, cache=False, pad=0.0, rect=False, rank=-1, world_size=1,
    workers=8, image_weights=False, quad=False, prefix=''):
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size, augment=
            augment, hyp=hyp, rect=rect, cache_images=cache, single_cls=opt
            .single_cls, stride=int(stride), pad=pad, image_weights=
            image_weights, prefix=prefix)
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else
        0, workers])
    sampler = torch.utils.data.distributed.DistributedSampler(dataset
        ) if rank != -1 else None
    loader = (torch.utils.data.DataLoader if image_weights else
        InfiniteDataLoader)
    dataloader = loader(dataset, batch_size=batch_size, num_workers=nw,
        sampler=sampler, pin_memory=True, collate_fn=LoadImagesAndLabels.
        collate_fn4 if quad else LoadImagesAndLabels.collate_fn)
    return dataloader, dataset
