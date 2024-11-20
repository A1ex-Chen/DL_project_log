def mpi_env_MPI_rank_and_size():
    """Get MPI rank and size from environment variables and return them as a
    tuple of integers.
    Most MPI implementations have an `mpirun` or `mpiexec` command that will
    run an MPI executable and set up all communication necessary between the
    different processors. As part of that set up, they will set environment
    variables that contain the rank and size of the MPI_COMM_WORLD
    communicator. We can read those environment variables from Python in order
    to ensure that `hvd.rank()` and `hvd.size()` return the expected values.
    Since MPI is just a standard, not an implementation, implementations
    typically choose their own environment variable names. This function tries
    to support several different implementation, but really it only needs to
    support whatever implementation we want to use for the TensorFlow test
    suite.
    If this is not running under MPI, then defaults of rank zero and size one
    are returned. (This is appropriate because when you call MPI_Init in an
    application not started with mpirun, it will create a new independent
    communicator with only one process in it.)

    Source: https://github.com/horovod/horovod/blob/c3626e/test/common.py#L25
    """
    rank_env = 'PMI_RANK SLURM_PROCID OMPI_COMM_WORLD_RANK'.split()
    size_env = 'PMI_SIZE SLURM_NTASKS OMPI_COMM_WORLD_SIZE'.split()
    for rank_var, size_var in zip(rank_env, size_env):
        rank = os.environ.get(rank_var)
        size = os.environ.get(size_var)
        if rank is not None and size is not None:
            return int(rank), int(size)
    return 0, 1
