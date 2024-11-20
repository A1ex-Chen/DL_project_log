def launch(fn, n_gpu_per_machine, n_machine=1, machine_rank=0, dist_url=
    None, args=()):
    world_size = n_machine * n_gpu_per_machine
    if world_size > 1:
        if 'OMP_NUM_THREADS' not in os.environ:
            os.environ['OMP_NUM_THREADS'] = '1'
        if dist_url == 'auto':
            if n_machine != 1:
                raise ValueError(
                    'dist_url="auto" not supported in multi-machine jobs')
            port = find_free_port()
            dist_url = f'tcp://127.0.0.1:{port}'
        if n_machine > 1 and dist_url.startswith('file://'):
            raise ValueError(
                'file:// is not a reliable init method in multi-machine jobs. Prefer tcp://'
                )
        mp.spawn(distributed_worker, nprocs=n_gpu_per_machine, args=(fn,
            world_size, n_gpu_per_machine, machine_rank, dist_url, args),
            daemon=False)
    else:
        fn(*args)
