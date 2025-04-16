def get_data_loader(dataset, batch_size, world_size, rank, shuffle=True,
    drop_last=True, num_workers=4, num_buckets=None):
    if world_size != 1:
        loader_shuffle = False
        if num_buckets:
            assert shuffle, 'only random buckets are supported'
            sampler = BucketingSampler(dataset, batch_size, num_buckets,
                world_size, rank)
            print('Using BucketingSampler')
        else:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            print('Using DistributedSampler')
    else:
        loader_shuffle = shuffle
        sampler = None
        print('Using no sampler')
    return DataLoader(batch_size=batch_size, drop_last=drop_last, sampler=
        sampler, shuffle=loader_shuffle, dataset=dataset, collate_fn=
        collate_fn, num_workers=num_workers, pin_memory=True)
