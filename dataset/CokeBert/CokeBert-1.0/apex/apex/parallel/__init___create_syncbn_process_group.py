def create_syncbn_process_group(group_size):
    """
    Creates process groups to be used for syncbn of a give ``group_size`` and returns
    process group that current GPU participates in.

    ``group_size`` must divide the total number of GPUs (world_size).

    ``group_size`` of 0 would be considered as =world_size. In this case ``None`` will be returned.

    ``group_size`` of 1 would be equivalent to using non-sync bn, but will still carry the overhead.

    Args:
        group_size (int): number of GPU's to collaborate for sync bn

    Example::

        >>> # model is an instance of torch.nn.Module
        >>> import apex
        >>> group = apex.parallel.create_syncbn_process_group(group_size)
    """
    if group_size == 0:
        return None
    world_size = torch.distributed.get_world_size()
    assert world_size >= group_size
    assert world_size % group_size == 0
    group = None
    for group_num in range(world_size // group_size):
        group_ids = range(group_num * group_size, (group_num + 1) * group_size)
        cur_group = torch.distributed.new_group(ranks=group_ids)
        if torch.distributed.get_rank() // group_size == group_num:
            group = cur_group
    assert group is not None
    return group
