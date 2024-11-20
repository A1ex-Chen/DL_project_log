def main():
    args = parse_args()
    args.amodel = args.amodel.replace('/', '-')
    pretrained_ckpts = sorted(glob.glob(os.path.join(args.pretrained,
        '*.pt')), key=os.path.getmtime)
    if args.name is None:
        args.name = '-'.join([datetime.now().strftime('%Y_%m_%d-%H_%M_%S'),
            f'linear_probemodel_{args.amodel}', f'lr_{args.lr}',
            f'b_{args.batch_size}', f'j_{args.workers}', f'p_{args.precision}']
            )
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    if args.remotedata and is_master(args):
        for dataset_name in args.datasetnames:
            for split in dataset_split[dataset_name]:
                if not os.path.exists(f'./json_files/{dataset_name}/{split}'):
                    os.makedirs(f'./json_files/{dataset_name}/{split}')
                os.system(
                    f'aws s3 cp s3://s-laion-audio/webdataset_tar/{dataset_name}/{split}/sizes.json ./json_files/{dataset_name}/{split}/sizes.json'
                    )
    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        postfix = 0
        while os.path.exists(args.log_path):
            postfix += 1
            log_base_path_new = log_base_path + '-' + str(postfix)
            os.makedirs(log_base_path_new, exist_ok=True)
            log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
            args.log_path = os.path.join(log_base_path_new, log_filename)
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)
    device = init_distributed_device(args)
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = ('tensorboard' in args.report_to or 'all' in args.
        report_to)
    if is_master(args):
        args.tensorboard_path = os.path.join(args.logs, args.name,
            'tensorboard') if args.tensorboard else ''
        args.checkpoint_path = os.path.join(args.logs, args.name, 'checkpoints'
            )
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''
        args.checkpoint_path = ''
    if args.copy_codebase:
        copy_codebase(args)
    assert args.precision in ['amp', 'fp16', 'fp32']
    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. FP16 support needs further verification and tuning, especially for train.'
            )
    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.'
            )
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.'
            )
    else:
        logging.info(f'Running with a single process. Device {args.device}.')
    logging.info(
        f'openai cache dir: {os.path.expanduser(args.openai_model_cache_dir)}')
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(
        args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, 'Please install tensorboard.'
        writer = tensorboard.SummaryWriter(args.tensorboard_path)
    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        wandb.init(project='clap', notes=args.wandb_notes, name=args.
            wandb_notes, tags=[], config=vars(args))
        logging.debug('Finished loading wandb.')
    for idx, f in enumerate(pretrained_ckpts):
        logging.info(f'pretrained on {f}')
        args.pretrained = f
        ckpt = torch.load(f, map_location='cpu')
        pretrain_epoch = 0
        if 'epoch' in ckpt:
            pretrain_epoch = ckpt['epoch']
        best_metrics = lp_main(args, device, writer, pretrain_epoch, idx)
        if args.wandb and is_master(args):
            assert wandb is not None, 'Please install wandb.'
            for name, val in best_metrics.items():
                wandb.log({f'val/summary/{name}': val, 'epoch': pretrain_epoch}
                    )
    if args.wandb and is_master(args):
        wandb.finish()
