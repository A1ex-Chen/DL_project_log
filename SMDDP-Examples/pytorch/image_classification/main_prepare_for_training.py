def prepare_for_training(args, model_args, model_arch):
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    else:
        args.local_rank = 0
    args.gpu = 0
    args.world_size = 1
    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='smddp', init_method='env://')
        args.world_size = dist.get_world_size()
    if args.seed is not None:
        print('Using seed = {}'.format(args.seed))
        torch.manual_seed(args.seed + args.local_rank)
        torch.cuda.manual_seed(args.seed + args.local_rank)
        np.random.seed(seed=args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

        def _worker_init_fn(id):
            np.random.seed(seed=args.seed + args.local_rank + id)
            random.seed(args.seed + args.local_rank + id)
    else:

        def _worker_init_fn(id):
            pass
    if args.static_loss_scale != 1.0:
        if not args.amp:
            print(
                'Warning: if --amp is not used, static_loss_scale will be ignored.'
                )
    if args.optimizer_batch_size < 0:
        batch_size_multiplier = 1
    else:
        tbs = args.world_size * args.batch_size
        args.optimizer_batch_size *= int(args.world_size / 8)
        if args.optimizer_batch_size % tbs != 0:
            print(
                'Warning: simulated batch size {} is not divisible by actual batch size {}'
                .format(args.optimizer_batch_size, tbs))
        batch_size_multiplier = int(args.optimizer_batch_size / tbs)
        print('BSM: {}'.format(batch_size_multiplier))
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda
                storage, loc: storage.cuda(args.gpu))
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model_state = checkpoint['state_dict']
            optimizer_state = checkpoint['optimizer']
            if 'state_dict_ema' in checkpoint:
                model_state_ema = checkpoint['state_dict_ema']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume,
                checkpoint['epoch']))
            if start_epoch >= args.epochs:
                print(
                    f'Launched training for {args.epochs}, checkpoint already run {start_epoch}'
                    )
                exit(1)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            model_state = None
            model_state_ema = None
            optimizer_state = None
    else:
        model_state = None
        model_state_ema = None
        optimizer_state = None
    loss = nn.CrossEntropyLoss
    if args.mixup > 0.0:
        loss = lambda : NLLMultiLabelSmooth(args.label_smoothing)
    elif args.label_smoothing > 0.0:
        loss = lambda : LabelSmoothing(args.label_smoothing)
    memory_format = (torch.channels_last if args.memory_format == 'nhwc' else
        torch.contiguous_format)
    model = model_arch(**{k: (v if k != 'pretrained' else v and (not args.
        distributed or dist.get_rank() == 0)) for k, v in model_args.
        __dict__.items()})
    image_size = (args.image_size if args.image_size is not None else model
        .arch.default_image_size)
    model_and_loss = ModelAndLoss(model, loss, cuda=True, memory_format=
        memory_format)
    if args.use_ema is not None:
        model_ema = deepcopy(model_and_loss)
        ema = EMA(args.use_ema)
    else:
        model_ema = None
        ema = None
    if args.data_backend == 'pytorch':
        get_train_loader = get_pytorch_train_loader
        get_val_loader = get_pytorch_val_loader
    elif args.data_backend == 'syntetic':
        get_val_loader = get_syntetic_loader
        get_train_loader = get_syntetic_loader
    else:
        print('Bad databackend picked')
        exit(1)
    train_loader, train_loader_len = get_train_loader(args.data, image_size,
        args.batch_size, model_args.num_classes, args.mixup > 0.0,
        interpolation=args.interpolation, augmentation=args.augmentation,
        start_epoch=start_epoch, workers=args.workers, memory_format=
        memory_format)
    if args.mixup != 0.0:
        train_loader = MixUpWrapper(args.mixup, train_loader)
    val_loader, val_loader_len = get_val_loader(args.data, image_size, args
        .batch_size, model_args.num_classes, False, interpolation=args.
        interpolation, workers=args.workers, memory_format=memory_format)
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger = log.Logger(args.print_freq, [dllogger.StdOutBackend(
            dllogger.Verbosity.DEFAULT, step_format=log.format_step),
            dllogger.JSONStreamBackend(dllogger.Verbosity.VERBOSE, os.path.
            join(args.workspace, args.raport_file))], start_epoch=
            start_epoch - 1)
    else:
        logger = log.Logger(args.print_freq, [], start_epoch=start_epoch - 1)
    logger.log_parameter(args.__dict__, verbosity=dllogger.Verbosity.DEFAULT)
    logger.log_parameter({f'model.{k}': v for k, v in model_args.__dict__.
        items()}, verbosity=dllogger.Verbosity.DEFAULT)
    optimizer = get_optimizer(list(model_and_loss.model.named_parameters()),
        args.lr, args=args, state=optimizer_state)
    if args.lr_schedule == 'step':
        lr_policy = lr_step_policy(args.lr, [30, 60, 80], 0.1, args.warmup,
            logger=logger)
    elif args.lr_schedule == 'cosine':
        lr_policy = lr_cosine_policy(args.lr, args.warmup, args.epochs,
            end_lr=args.end_lr, logger=logger)
    elif args.lr_schedule == 'linear':
        lr_policy = lr_linear_policy(args.lr, args.warmup, args.epochs,
            logger=logger)
    scaler = torch.cuda.amp.GradScaler(init_scale=args.static_loss_scale,
        growth_factor=2, backoff_factor=0.5, growth_interval=100 if args.
        dynamic_loss_scale else 1000000000, enabled=args.amp)
    if args.distributed:
        model_and_loss.distributed(args.gpu)
    model_and_loss.load_model_state(model_state)
    if ema is not None and model_state_ema is not None:
        print('load ema')
        ema.load_state_dict(model_state_ema)
    return (model_and_loss, optimizer, lr_policy, scaler, train_loader,
        val_loader, logger, ema, model_ema, train_loader_len,
        batch_size_multiplier, start_epoch)
