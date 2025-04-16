def get_world_size():
    if not is_distributed():
        return 1
    return dist.get_world_size()
