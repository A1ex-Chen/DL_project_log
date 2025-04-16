def all_gather_arbitary_tensor(tensor):
    if get_world_size() > 1:
        device = tensor.device
        tensor_batch = all_gather_pickle(tensor.cpu(), device)
        tensor_batch = [x.to(device) for x in tensor_batch]
        tensor_batch[torch.distributed.get_rank()] = tensor
        tensor_batch = torch.cat(tensor_batch, dim=0)
    else:
        tensor_batch = tensor
    return tensor_batch
