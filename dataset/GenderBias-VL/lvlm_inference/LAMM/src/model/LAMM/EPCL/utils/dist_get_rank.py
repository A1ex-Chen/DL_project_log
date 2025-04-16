def get_rank():
    if not is_distributed():
        return 0
    return dist.get_rank()
