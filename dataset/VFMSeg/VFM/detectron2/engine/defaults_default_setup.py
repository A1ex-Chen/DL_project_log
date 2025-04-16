def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = _try_get_key(cfg, 'OUTPUT_DIR', 'output_dir',
        'train.output_dir')
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)
    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name='fvcore')
    logger = setup_logger(output_dir, distributed_rank=rank)
    logger.info('Rank of current process: {}. World size: {}'.format(rank,
        comm.get_world_size()))
    logger.info('Environment info:\n' + collect_env_info())
    logger.info('Command line arguments: ' + str(args))
    if hasattr(args, 'config_file') and args.config_file != '':
        logger.info('Contents of args.config_file={}:\n{}'.format(args.
            config_file, _highlight(PathManager.open(args.config_file, 'r')
            .read(), args.config_file)))
    if comm.is_main_process() and output_dir:
        path = os.path.join(output_dir, 'config.yaml')
        if isinstance(cfg, CfgNode):
            logger.info('Running with full config:\n{}'.format(_highlight(
                cfg.dump(), '.yaml')))
            with PathManager.open(path, 'w') as f:
                f.write(cfg.dump())
        else:
            LazyConfig.save(cfg, path)
        logger.info('Full config saved to {}'.format(path))
    seed = _try_get_key(cfg, 'SEED', 'train.seed', default=-1)
    seed_all_rng(None if seed < 0 else seed + rank)
    if not (hasattr(args, 'eval_only') and args.eval_only):
        torch.backends.cudnn.benchmark = _try_get_key(cfg,
            'CUDNN_BENCHMARK', 'train.cudnn_benchmark', default=False)
