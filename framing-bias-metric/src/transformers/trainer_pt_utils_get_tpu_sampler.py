def get_tpu_sampler(dataset: torch.utils.data.dataset.Dataset):
    if xm.xrt_world_size() <= 1:
        return RandomSampler(dataset)
    return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal())
