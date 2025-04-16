def create_classification_dataloader(path, imgsz=224, batch_size=16,
    augment=True, cache=False, rank=-1, workers=8, shuffle=True):
    with torch_distributed_zero_first(rank):
        dataset = ClassificationDataset(root=path, imgsz=imgsz, augment=
            augment, cache=cache)
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else
        0, workers])
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset,
        shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(dataset, batch_size=batch_size, shuffle=
        shuffle and sampler is None, num_workers=nw, sampler=sampler,
        pin_memory=PIN_MEMORY, worker_init_fn=seed_worker, generator=generator)
