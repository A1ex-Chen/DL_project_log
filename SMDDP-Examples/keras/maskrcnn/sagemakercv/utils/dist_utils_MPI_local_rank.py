def MPI_local_rank():
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK'))
    elif 'SLURM_LOCALID' in os.environ:
        return int(os.environ.get('SLURM_LOCALID'))
    else:
        return 0
