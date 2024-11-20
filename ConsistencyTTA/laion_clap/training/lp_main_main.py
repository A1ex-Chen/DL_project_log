def main():
    args = parse_args()
    time.sleep(args.sleep)
    args.amodel = args.amodel.replace('/', '-')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    args.class_index_dict = load_class_label(args.class_label_path)
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
    clap_model, clap_model_cfg = create_model(args.amodel, args.tmodel,
        args.pretrained, precision=args.precision, device=device, jit=args.
        torchscript, force_quick_gelu=args.force_quick_gelu,
        openai_model_cache_dir=os.path.expanduser(args.
        openai_model_cache_dir), skip_params=False, pretrained_audio=args.
        pretrained_audio, pretrained_text=args.pretrained_text,
        enable_fusion=args.enable_fusion, fusion_type=args.fusion_type)
    args.lp_out_ch = len(list(args.class_index_dict.keys()))
    logging.info(f'linear probe using mlp: {args.lp_mlp}')
    logging.info(f'linear probe using freeze: {args.lp_freeze}')
    logging.info(f'linear probe act layer: {args.lp_act}')
    logging.info(f'linear probe out ch: {args.lp_out_ch}')
    logging.info(f'linear probe learning rate (if applicable): {args.lp_lr}')
    logging.info(f'linear probe loss func: {args.lp_loss}')
    logging.info(f'linear probe lp_metrics: {args.lp_metrics}')
    model = LinearProbe(clap_model, mlp=args.lp_mlp, freeze=args.lp_freeze,
        in_ch=512, out_ch=args.lp_out_ch, act=args.lp_act)
    model = model.to(device)
    if args.horovod:
        with torch.no_grad():
            for param in model.parameters():
                param.set_(param.contiguous())
    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)
    if is_master(args):
        logging.info('Linear Probe CLAP Model:')
        logging.info(f'{str(clap_model)}')
        logging.info('Params:')
        params_file = os.path.join(args.logs, args.name, 'params.txt')
        with open(params_file, 'w') as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f'  {name}: {val}')
                f.write(f'{name}: {val}\n')
    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids
            =[device], find_unused_parameters=True, **ddp_args)
    data = get_data(args, clap_model_cfg)
    assert len(data), 'At least one train or eval dataset must be specified.'
    if args.trace:
        assert 'train' not in data, 'Cannot train with traced model'
    optimizer, scheduler, text_freeze_parameters = config_lp_optimizer(model,
        data, args)
    scaler = GradScaler() if args.precision == 'amp' else None
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                sd = checkpoint['state_dict']
                if not args.distributed and next(iter(sd.items()))[0
                    ].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                if args.split_opt:
                    if optimizer is not None:
                        for k, o_ in optimizer.items():
                            o_.load_state_dict(checkpoint[k + '_' +
                                'optimizer'])
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                if scaler is not None and 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                logging.info(
                    f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})"
                    )
            else:
                model.load_state_dict(checkpoint)
                logging.info(
                    f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})"
                    )
            if args.freeze_text:
                print('Freeze Text!!!!')
                for k in text_freeze_parameters:
                    k.requires_grad = False
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True
    cudnn.deterministic = False
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(
        args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, 'Please install tensorboard.'
        writer = tensorboard.SummaryWriter(args.tensorboard_path)
    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data['train'].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data['val'].dataloader.num_samples
        wandb.init(project='clap', notes=args.wandb_notes, name=args.
            wandb_notes, tags=[], config=vars(args))
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')
    if 'train' not in data:
        evaluate(model, data, start_epoch, args, writer)
        return
    elif start_epoch == 0 and 'val' in data and not args.no_eval:
        evaluate(model, data, 0, args, writer)
    if args.save_top_performance:
        current_top_k_ckpt_metrics = {i: (0) for i in range(args.
            save_top_performance)}
    for epoch in range(start_epoch, args.epochs):
        if epoch == args.freeze_text_after:
            print('Text pretrained parameters are freezed since this epoch.')
            for k in text_freeze_parameters:
                k.requires_grad = False
        if is_master(args):
            logging.info(f'Start epoch {epoch}')
        train_one_epoch(model, data, epoch, optimizer, scaler, scheduler,
            args, writer)
        completed_epoch = epoch + 1
        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')
            ) and not args.no_eval:
            metrics = evaluate(model, data, completed_epoch, args, writer)
            if args.save_top_performance:
                top_k_dataset = args.top_k_checkpoint_select_dataset
                top_k_metric = args.top_k_checkpoint_select_metric
                filtered_metrics = [v for k, v in metrics.items() if 
                    top_k_metric in k and top_k_dataset in k]
        if args.save_logs:
            opt_dict = {(k + '_' + 'optimizer'): v.state_dict() for k, v in
                optimizer.items()}
            checkpoint_dict = {'epoch': completed_epoch, 'name': args.name,
                'state_dict': model.state_dict()}
            checkpoint_dict.update(opt_dict)
            if scaler is not None:
                checkpoint_dict['scaler'] = scaler.state_dict()
            if (completed_epoch == args.epochs or args.save_frequency > 0 and
                completed_epoch % args.save_frequency == 0):
                torch.save(checkpoint_dict, os.path.join(args.
                    checkpoint_path, f'epoch_{completed_epoch}.pt'))
            if args.save_most_recent:
                torch.save(checkpoint_dict, os.path.join(args.
                    checkpoint_path, f'epoch_latest.pt'))
            if args.save_top_performance and not args.no_eval:
                update_top_k_performance(filtered_metrics,
                    current_top_k_ckpt_metrics, args, checkpoint_dict,
                    bignumbetter=True)
    if args.wandb and is_master(args):
        wandb.finish()
