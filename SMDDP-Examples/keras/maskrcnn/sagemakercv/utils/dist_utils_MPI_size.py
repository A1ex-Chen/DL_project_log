def MPI_size():
    return int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
