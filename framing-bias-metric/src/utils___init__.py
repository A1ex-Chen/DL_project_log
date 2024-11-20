def __init__(self, dataset, batch_size, num_replicas=None, rank=None,
    add_extra_examples=True, shuffle=True):
    if num_replicas is None:
        if not dist.is_available():
            raise RuntimeError('Requires distributed package to be available')
        num_replicas = dist.get_world_size()
    if rank is None:
        if not dist.is_available():
            raise RuntimeError('Requires distributed package to be available')
        rank = dist.get_rank()
    self.dataset = dataset
    self.num_replicas = num_replicas
    self.rank = rank
    self.epoch = 0
    if add_extra_examples:
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.
            num_replicas))
        self.total_size = self.num_samples * self.num_replicas
    else:
        self.total_size = len(dataset)
        self.num_samples = len(self.available_indices)
    self.batch_size = batch_size
    self.add_extra_examples = add_extra_examples
    self.shuffle = shuffle
