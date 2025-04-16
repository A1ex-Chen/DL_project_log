def all_reduce_average(tensor):
    val = all_reduce_sum(tensor)
    return val / get_world_size()
