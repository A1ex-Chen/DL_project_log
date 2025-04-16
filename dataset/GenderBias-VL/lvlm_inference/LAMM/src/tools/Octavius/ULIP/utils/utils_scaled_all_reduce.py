def scaled_all_reduce(tensors, is_scale=True):
    """Performs the scaled all_reduce operation on the provided tensors.
    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    world size.
    """
    world_size = get_world_size()
    if world_size == 1:
        return tensors
    reductions = []
    for tensor in tensors:
        reduction = dist.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    for reduction in reductions:
        reduction.wait()
    if is_scale:
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors
