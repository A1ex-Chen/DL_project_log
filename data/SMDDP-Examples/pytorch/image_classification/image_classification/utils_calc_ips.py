def calc_ips(batch_size, time):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    tbs = world_size * batch_size
    return tbs / time
