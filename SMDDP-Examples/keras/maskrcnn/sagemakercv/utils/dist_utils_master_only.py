def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank = MPI_rank()
        if rank == 0:
            return func(*args, **kwargs)
    return wrapper