def init_distributed_device(args):
    args.distributed = False
    args.world_size = 1
    args.rank = 0
    args.local_rank = 0
    if args.horovod:
        assert hvd is not None, 'Horovod is not installed'
        hvd.init()
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.local_rank = local_rank
        args.rank = world_rank
        args.world_size = world_size
        args.distributed = True
        os.environ['LOCAL_RANK'] = str(args.local_rank)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        print(
            f'Distributed training: local_rank={args.local_rank}, rank={args.rank}, world_size={args.world_size}, hostname={socket.gethostname()}, pid={os.getpid()}'
            )
    elif is_using_distributed():
        if 'SLURM_PROCID' in os.environ:
            args.local_rank, args.rank, args.world_size = world_info_from_env()
            os.environ['LOCAL_RANK'] = str(args.local_rank)
            os.environ['RANK'] = str(args.rank)
            os.environ['WORLD_SIZE'] = str(args.world_size)
            torch.distributed.init_process_group(backend=args.dist_backend,
                init_method=args.dist_url, world_size=args.world_size, rank
                =args.rank)
        elif 'OMPI_COMM_WORLD_SIZE' in os.environ:
            world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
            world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
            local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
            args.local_rank = local_rank
            args.rank = world_rank
            args.world_size = world_size
            torch.distributed.init_process_group(backend=args.dist_backend,
                init_method=args.dist_url, world_size=args.world_size, rank
                =args.rank)
        else:
            args.local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(backend=args.dist_backend,
                init_method=args.dist_url)
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()
        args.distributed = True
        print(
            f'Distributed training: local_rank={args.local_rank}, rank={args.rank}, world_size={args.world_size}, hostname={socket.gethostname()}, pid={os.getpid()}'
            )
    if torch.cuda.is_available():
        if args.distributed and not args.no_set_device_rank:
            device = 'cuda:%d' % args.local_rank
        else:
            device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    args.device = device
    device = torch.device(device)
    return device
