def all_reduce(tensor, op=dist.ReduceOp.SUM):
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=op)
    return tensor
