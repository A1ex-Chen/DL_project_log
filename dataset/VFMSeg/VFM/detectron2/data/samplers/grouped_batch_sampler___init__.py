def __init__(self, sampler, group_ids, batch_size):
    """
        Args:
            sampler (Sampler): Base sampler.
            group_ids (list[int]): If the sampler produces indices in range [0, N),
                `group_ids` must be a list of `N` ints which contains the group id of each sample.
                The group ids must be a set of integers in the range [0, num_groups).
            batch_size (int): Size of mini-batch.
        """
    if not isinstance(sampler, Sampler):
        raise ValueError(
            'sampler should be an instance of torch.utils.data.Sampler, but got sampler={}'
            .format(sampler))
    self.sampler = sampler
    self.group_ids = np.asarray(group_ids)
    assert self.group_ids.ndim == 1
    self.batch_size = batch_size
    groups = np.unique(self.group_ids).tolist()
    self.buffer_per_group = {k: [] for k in groups}
