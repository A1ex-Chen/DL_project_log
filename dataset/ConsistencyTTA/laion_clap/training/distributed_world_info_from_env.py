def world_info_from_env():
    local_rank = 0
    for v in ('SLURM_LOCALID', 'MPI_LOCALRANKID',
        'OMPI_COMM_WORLD_LOCAL_RANK', 'LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('SLURM_PROCID', 'PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('SLURM_NTASKS', 'PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'WORLD_SIZE'
        ):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    return local_rank, global_rank, world_size
