def _distributed_worker(local_rank, main_func, world_size,
    num_gpus_per_machine, machine_rank, dist_url, args, timeout=DEFAULT_TIMEOUT
    ):
    assert torch.cuda.is_available(
        ), 'cuda is not available. Please check your installation.'
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(backend='NCCL', init_method=dist_url,
            world_size=world_size, rank=global_rank, timeout=timeout)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error('Process group URL: {}'.format(dist_url))
        raise e
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) *
            num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg
    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    comm.synchronize()
    main_func(*args)
