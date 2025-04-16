def init_distributed(cuda):
    """
    Initializes distributed backend.

    :param cuda: (bool) if True initializes nccl backend, if False initializes
        gloo backend
    """
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    distributed = world_size > 1
    if distributed:
        backend = 'nccl' if cuda else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')
        assert dist.is_initialized()
    return distributed
