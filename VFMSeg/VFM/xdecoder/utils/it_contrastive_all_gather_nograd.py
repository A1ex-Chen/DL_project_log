@torch.no_grad()
def all_gather_nograd(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if get_world_size() > 1:
        tensors_gather = [torch.ones_like(tensor) for _ in range(torch.
            distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        tensor = torch.cat(tensors_gather, dim=0)
    return tensor
