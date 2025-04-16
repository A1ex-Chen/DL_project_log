def main():
    args = parse_args()
    from xmuda.common.config import purge_cfg
    from xmuda.config.xmuda import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace(
            '../configs/', ''))
        if not osp.isdir(output_dir):
            warnings.warn('Make a new directory: {}'.format(output_dir))
            os.makedirs(output_dir)
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)
    logger = setup_logger('xmuda', output_dir, comment='test.{:s}'.format(
        run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)
    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))
    assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD
    test(cfg, args, output_dir)
