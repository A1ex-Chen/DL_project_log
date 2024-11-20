def all_gather_grad(x):
    if get_world_size() > 1:
        all_x = [torch.zeros_like(x) for _ in range(get_world_size())]
        torch.distributed.all_gather(all_x, x)
        all_x[torch.distributed.get_rank()] = x
        x = torch.cat(all_x, dim=0)
    return x
