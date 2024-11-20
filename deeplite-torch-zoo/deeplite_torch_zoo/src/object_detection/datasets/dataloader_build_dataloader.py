def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers]
        )
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset,
        shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return dataloader.DataLoader(dataset=dataset, batch_size=batch, shuffle
        =shuffle and sampler is None, num_workers=nw, sampler=sampler,
        pin_memory=PIN_MEMORY, collate_fn=getattr(dataset, 'collate_fn',
        None), worker_init_fn=seed_worker, generator=generator)
