def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return tensors
    tensor_all = GatherLayer.apply(tensors)
    return torch.cat(tensor_all, dim=0)
