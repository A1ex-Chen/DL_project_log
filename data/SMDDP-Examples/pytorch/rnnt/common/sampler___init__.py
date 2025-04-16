def __init__(self, dataset, batch_size, num_buckets, world_size, rank):
    """
        Bucketing sampler with approx. equally-sized buckets.
        :param dataset: dataset
        :param batch_size: local batch size
        :param seeds: list of seeds, one seed for each training epoch
        :param num_buckets: number of buckets
        :param world_size: number of distributed workers
        :param rank: rank of the current process
        """
    super().__init__(dataset, batch_size, world_size, rank)
    self.num_buckets = num_buckets
    len_ids = np.argsort([sample['duration'] for sample in dataset.samples])
    self.buckets = [torch.from_numpy(t) for t in np.array_split(len_ids,
        num_buckets)]
    global_bs = self.global_batch_size
