def get_local_rank() ->int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None, 'Local process group is not created! Please use launch() to spawn processes!'
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)
