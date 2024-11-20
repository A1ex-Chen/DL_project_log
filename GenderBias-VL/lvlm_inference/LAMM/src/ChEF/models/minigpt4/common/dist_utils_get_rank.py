def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()
