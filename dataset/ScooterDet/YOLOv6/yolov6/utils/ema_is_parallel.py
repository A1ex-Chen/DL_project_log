def is_parallel(model):
    """Return True if model's type is DP or DDP, else False."""
    return type(model) in (nn.parallel.DataParallel, nn.parallel.
        DistributedDataParallel)
