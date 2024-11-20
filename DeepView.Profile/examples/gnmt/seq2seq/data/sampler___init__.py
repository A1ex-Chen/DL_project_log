def __init__(self, dataset, batch_size, pad, world_size=None, rank=None):
    """
        Constructor for the StaticDistributedSampler.

        :param dataset: dataset
        :param batch_size: local batch size
        :param pad: if True: pads dataset to a multiple of global_batch_size
            samples
        :param world_size: number of distributed workers
        :param rank: rank of the current process
        """
    if world_size is None:
        world_size = get_world_size()
    if rank is None:
        rank = get_rank()
    self.world_size = world_size
    global_batch_size = batch_size * world_size
    data_len = len(dataset)
    num_samples = (data_len + global_batch_size - 1
        ) // global_batch_size * global_batch_size
    self.num_samples = num_samples
    indices = list(range(data_len))
    if pad:
        indices += [0] * (num_samples - len(indices))
    else:
        indices += [-1] * (num_samples - len(indices))
    indices = torch.tensor(indices)
    indices = indices.view(-1, batch_size)
    indices = indices[rank::world_size].contiguous()
    indices = indices.view(-1)
    indices = indices[indices != -1]
    indices = indices.tolist()
    self.indices = indices
