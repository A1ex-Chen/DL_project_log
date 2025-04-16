def get_dist_info():
    rank = MPI_rank()
    local_rank = MPI_local_rank()
    size = MPI_size()
    local_size = MPI_local_size()
    return rank, local_rank, size, local_size
