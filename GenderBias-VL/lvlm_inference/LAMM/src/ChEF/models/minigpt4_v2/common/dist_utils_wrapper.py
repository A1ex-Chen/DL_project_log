@functools.wraps(func)
def wrapper(*args, **kwargs):
    rank, _ = get_dist_info()
    if rank == 0:
        return func(*args, **kwargs)
