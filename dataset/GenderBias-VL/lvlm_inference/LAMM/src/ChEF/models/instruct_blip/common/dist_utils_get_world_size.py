def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
