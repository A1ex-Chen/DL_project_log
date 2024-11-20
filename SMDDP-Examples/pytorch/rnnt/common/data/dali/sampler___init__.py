def __init__(self, config_data, num_buckets, batch_size, num_workers,
    num_epochs, seed, dist_sampler, pre_sort):
    super(VectorizedBucketingSampler, self).__init__(config_data, dist_sampler)
    self.seed = seed
    self.num_buckets = num_buckets
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.pre_sort = pre_sort
