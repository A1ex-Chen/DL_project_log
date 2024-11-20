def MPI_rank():
    return int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
