def get_local_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    if LOCAL_PROCESS_GROUP is None:
        raise ValueError('tensorfn.distributed.LOCAL_PROCESS_GROUP is None')
    return dist.get_rank(group=LOCAL_PROCESS_GROUP)
