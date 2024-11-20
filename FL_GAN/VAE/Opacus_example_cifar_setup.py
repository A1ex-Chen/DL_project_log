def setup(args):
    if not torch.cuda.is_available():
        raise NotImplementedError(
            'DistributedDataParallel device_ids and output_device arguments             only work with single-device GPU modules'
            )
    if sys.platform == 'win32':
        raise NotImplementedError(
            'Windows version of multi-GPU is not supported yet.')
    if os.environ.get('SLURM_NTASKS') is not None:
        rank = int(os.environ.get('SLURM_PROCID'))
        local_rank = int(os.environ.get('SLURM_LOCALID'))
        world_size = int(os.environ.get('SLURM_NTASKS'))
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '7440'
        torch.distributed.init_process_group(args.dist_backend, rank=rank,
            world_size=world_size)
        logger.debug(
            f'Setup on Slurm: rank={rank}, local_rank={local_rank}, world_size={world_size}'
            )
        return rank, local_rank, world_size
    elif args.local_rank >= 0:
        torch.distributed.init_process_group(init_method='env://', backend=
            args.dist_backend)
        rank = torch.distributed.get_rank()
        local_rank = args.local_rank
        world_size = torch.distributed.get_world_size()
        logger.debug(
            f"Setup with 'env://': rank={rank}, local_rank={local_rank}, world_size={world_size}"
            )
        return rank, local_rank, world_size
    else:
        logger.debug(f'Running on a single GPU.')
        return 0, 0, 1
