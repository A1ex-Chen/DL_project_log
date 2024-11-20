def launch(main_func, num_gpus_per_machine, num_machines=1, machine_rank=0,
    dist_url=None, args=(), timeout=DEFAULT_TIMEOUT):
    """
    Launch multi-gpu or distributed training.
    This function must be called on all machines involved in the training.
    It will spawn child processes (defined by ``num_gpus_per_machine``) on each machine.

    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_gpus_per_machine (int): number of GPUs per machine
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine
        dist_url (str): url to connect to for distributed jobs, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to "auto" to automatically select a free port on localhost
        timeout (timedelta): timeout of the distributed workers
        args (tuple): arguments passed to main_func
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        if dist_url == 'auto':
            assert num_machines == 1, 'dist_url=auto not supported in multi-machine jobs.'
            port = _find_free_port()
            dist_url = f'tcp://127.0.0.1:{port}'
        if num_machines > 1 and dist_url.startswith('file://'):
            logger = logging.getLogger(__name__)
            logger.warning(
                'file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://'
                )
        mp.spawn(_distributed_worker, nprocs=num_gpus_per_machine, args=(
            main_func, world_size, num_gpus_per_machine, machine_rank,
            dist_url, args, timeout), daemon=False)
    else:
        main_func(*args)
