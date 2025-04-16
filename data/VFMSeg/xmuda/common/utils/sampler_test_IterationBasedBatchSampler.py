def test_IterationBasedBatchSampler():
    from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
    sampler = RandomSampler([i for i in range(9)])
    batch_sampler = BatchSampler(sampler, batch_size=2, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, 6, start_iter=0)
    for i, index in enumerate(batch_sampler):
        print(i, index)
