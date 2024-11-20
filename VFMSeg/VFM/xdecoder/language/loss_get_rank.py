def get_rank():
    if is_dist_initialized():
        return dist.get_rank()
    return 0
