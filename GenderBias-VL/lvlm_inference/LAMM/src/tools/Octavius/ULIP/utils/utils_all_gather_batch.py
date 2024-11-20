def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    world_size = get_world_size()
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_all, tensor, async_op=False)
        tensor_list.append(tensor_all)
    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor
