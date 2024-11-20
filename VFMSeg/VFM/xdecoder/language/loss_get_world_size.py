def get_world_size():
    if is_dist_initialized():
        return torch.distributed.get_world_size()
    return 1
