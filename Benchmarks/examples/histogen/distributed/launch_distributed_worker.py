def distributed_worker(local_rank, fn, world_size, n_gpu_per_machine,
    machine_rank, dist_url, args):
    if not torch.cuda.is_available():
        raise OSError('CUDA is not available. Please check your environments')
    global_rank = machine_rank * n_gpu_per_machine + local_rank
    try:
        dist.init_process_group(backend='NCCL', init_method=dist_url,
            world_size=world_size, rank=global_rank)
    except Exception:
        raise OSError('failed to initialize NCCL groups')
    dist_fn.synchronize()
    if n_gpu_per_machine > torch.cuda.device_count():
        raise ValueError(
            f'specified n_gpu_per_machine larger than available device ({torch.cuda.device_count()})'
            )
    torch.cuda.set_device(local_rank)
    if dist_fn.LOCAL_PROCESS_GROUP is not None:
        raise ValueError('torch.distributed.LOCAL_PROCESS_GROUP is not None')
    n_machine = world_size // n_gpu_per_machine
    for i in range(n_machine):
        ranks_on_i = list(range(i * n_gpu_per_machine, (i + 1) *
            n_gpu_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            dist_fn.distributed.LOCAL_PROCESS_GROUP = pg
    fn(*args)
