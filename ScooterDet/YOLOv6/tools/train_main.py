def main(args):
    """main function of training"""
    args.local_rank, args.rank, args.world_size = get_envs()
    cfg, device, args = check_and_init(args)
    args.local_rank, args.rank, args.world_size = get_envs()
    LOGGER.info(f'training args are: {args}\n')
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        LOGGER.info('Initializing process group... ')
        dist.init_process_group(backend='nccl' if dist.is_nccl_available() else
            'gloo', init_method=args.dist_url, rank=args.local_rank,
            world_size=args.world_size, timeout=datetime.timedelta(seconds=
            7200))
    trainer = Trainer(args, cfg, device)
    if args.quant and args.calib:
        trainer.calibrate(cfg)
        return
    trainer.train()
    if args.world_size > 1 and args.rank == 0:
        LOGGER.info('Destroying process group... ')
        dist.destroy_process_group()
