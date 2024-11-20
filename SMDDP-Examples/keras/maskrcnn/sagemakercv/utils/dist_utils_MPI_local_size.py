def MPI_local_size():
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', 1))
