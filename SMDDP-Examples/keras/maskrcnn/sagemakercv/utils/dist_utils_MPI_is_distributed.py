def MPI_is_distributed():
    """Return a boolean whether a distributed training/inference runtime is being used.
    :return: bool
    """
    if all([(var in os.environ) for var in ['OMPI_COMM_WORLD_RANK',
        'OMPI_COMM_WORLD_SIZE']]):
        return True
    elif all([(var in os.environ) for var in ['SLURM_PROCID', 'SLURM_NTASKS']]
        ):
        return True
    else:
        return False
