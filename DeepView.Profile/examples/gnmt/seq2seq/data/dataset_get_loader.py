def get_loader(self, batch_size=1, seeds=None, shuffle=False, num_workers=0,
    batch_first=False, pad=False, batching=None, batching_opt={}):
    collate_fn = build_collate_fn(batch_first, parallel=self.parallel, sort
        =True)
    if shuffle:
        if batching == 'random':
            sampler = DistributedSampler(self, batch_size, seeds)
        elif batching == 'sharding':
            sampler = ShardingSampler(self, batch_size, seeds, batching_opt
                ['shard_size'])
        elif batching == 'bucketing':
            sampler = BucketingSampler(self, batch_size, seeds,
                batching_opt['num_buckets'])
        else:
            raise NotImplementedError
    else:
        sampler = StaticDistributedSampler(self, batch_size, pad)
    return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn,
        sampler=sampler, num_workers=num_workers, pin_memory=True,
        drop_last=False)
