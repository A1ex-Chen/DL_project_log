def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.
        DistributedDataParallel)
