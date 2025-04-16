def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    args.prefetcher = not args.no_prefetcher
    args.grad_accum_steps = max(1, args.grad_accum_steps)
    device = utils.init_distributed_device(args)
    if args.distributed:
        _logger.info(
            f'Training in distributed mode with multiple processes, 1 device per process.Process {args.rank}, total {args.world_size}, device {args.device}.'
            )
    else:
        _logger.info(
            f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            use_amp = 'apex'
            assert args.amp_dtype == 'float16'
        else:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            use_amp = 'native'
            assert args.amp_dtype in ('float16', 'bfloat16')
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16
    utils.random_seed(args.seed, args.rank)
    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()
    model_kd = None
    if args.kd_model_name is not None:
        model_kd = KDTeacher(args)
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]
    model_kwargs = {'in_chans': in_chans, 'drop_rate': args.drop,
        'drop_path_rate': args.drop_path, 'drop_block_rate': args.
        drop_block, 'global_pool': args.gp, 'bn_momentum': args.bn_momentum,
        'bn_eps': args.bn_eps, 'scriptable': args.torchscript,
        'checkpoint_path': args.initial_checkpoint, **args.model_kwargs}
    model = get_model(model_name=args.model, dataset_name=args.
        pretraining_dataset, pretrained=args.pretrained, num_classes=args.
        num_classes, **model_kwargs)
    if hasattr(model, 'get_classifier'):
        if args.head_init_scale is not None:
            with torch.no_grad():
                model.get_classifier().weight.mul_(args.head_init_scale)
                model.get_classifier().bias.mul_(args.head_init_scale)
        if args.head_init_bias is not None:
            nn.init.constant_(model.get_classifier().bias, args.head_init_bias)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'
            ), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes
    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)
    if utils.is_primary(args):
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}'
            )
    data_config = resolve_data_config(vars(args), model=model, verbose=
        utils.is_primary(args))
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))
    model.to(device=device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    if args.distributed and args.sync_bn:
        args.dist_bn = ''
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            model = convert_syncbn_model(model)
        else:
            model = convert_sync_batchnorm(model)
        if utils.is_primary(args):
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.'
                )
    if args.torchscript:
        assert not args.torchcompile
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)
    if not args.lr:
        global_batch_size = (args.batch_size * args.world_size * args.
            grad_accum_steps)
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = 'sqrt' if any([(o in on) for o in ('ada',
                'lamb')]) else 'linear'
        if args.lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f'Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling.'
                )
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args), **
        args.opt_kwargs)
    amp_autocast = suppress
    loss_scaler = None
    if use_amp == 'apex':
        assert device.type == 'cuda'
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if utils.is_primary(args):
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        try:
            amp_autocast = partial(torch.autocast, device_type=device.type,
                dtype=amp_dtype)
        except (AttributeError, TypeError):
            assert device.type == 'cuda'
            amp_autocast = torch.cuda.amp.autocast
        if device.type == 'cuda' and amp_dtype == torch.float16:
            loss_scaler = NativeScaler()
        if utils.is_primary(args):
            _logger.info('Using native Torch AMP. Training in mixed precision.'
                )
    elif utils.is_primary(args):
        _logger.info('AMP not enabled. Training in float32.')
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(model, args.resume, optimizer=None if
            args.no_resume_opt else optimizer, loss_scaler=None if args.
            no_resume_opt else loss_scaler, log_info=utils.is_primary(args))
    model_ema = None
    if args.model_ema:
        model_ema = utils.ModelEmaV2(model, decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)
    if args.distributed:
        if has_apex and use_amp == 'apex':
            if utils.is_primary(args):
                _logger.info('Using NVIDIA APEX DistributedDataParallel.')
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if utils.is_primary(args):
                _logger.info('Using native Torch DistributedDataParallel.')
            model = NativeDDP(model, device_ids=[device], broadcast_buffers
                =not args.no_ddp_bb)
    if args.torchcompile:
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        model = torch.compile(model, backend=args.torchcompile)
    if args.data and not args.data_dir:
        args.data_dir = args.data
    collate_fn = None
    mixup_fn = None
    mixup_active = (args.mixup > 0 or args.cutmix > 0.0 or args.
        cutmix_minmax is not None)
    if mixup_active:
        mixup_args = dict(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax, prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    extra_loader_kwargs = {}
    if args.dataset == 'imagenet':
        extra_loader_kwargs = {'train_split': args.train_split, 'val_split':
            args.val_split, 'class_map': args.class_map, 'seed': args.seed,
            'repeats': args.epoch_repeats}
    dataloaders = get_dataloaders(dataset_name=args.dataset, data_root=args
        .data_dir, img_size=data_config['input_size'], batch_size=args.
        batch_size, test_batch_size=args.validation_batch_size or args.
        batch_size, download=args.dataset_download, distributed=args.
        distributed, use_prefetcher=args.prefetcher, no_aug=args.no_aug,
        re_prob=args.reprob, re_mode=args.remode, re_count=args.recount,
        re_split=args.resplit, scale=args.scale, ratio=args.ratio, hflip=
        args.hflip, vflip=args.vflip, color_jitter=args.color_jitter,
        auto_augment=args.aa, num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits, train_interpolation=
        train_interpolation, test_interpolation=data_config['interpolation'
        ], mean=data_config['mean'], std=data_config['std'], num_workers=
        args.workers, collate_fn=collate_fn, pin_memory=args.pin_mem,
        device=device, use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding, **extra_loader_kwargs)
    loader_train, loader_eval = dataloaders['train'], dataloaders['test']
    dataset_train = loader_train.dataset
    if args.jsd_loss:
        assert num_aug_splits > 1
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits,
            smoothing=args.smoothing)
    elif mixup_active:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.
                bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing,
                target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing
                )
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.to(device=device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([datetime.now().strftime('%Y%m%d-%H%M%S'),
                safe_model_name(args.model), str(data_config['input_size'][
                -1])])
        output_dir = utils.get_outdir(args.output if args.output else
            './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(model=model, optimizer=optimizer, args=args,
            model_ema=model_ema, amp_scaler=loss_scaler, checkpoint_dir=
            output_dir, recovery_dir=output_dir, decreasing=decreasing,
            max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
        if args.log_tb:
            writer = SummaryWriter(log_dir=output_dir)
    if utils.is_primary(args) and args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. Metrics not being logged to wandb, try `pip install wandb`"
                )
    updates_per_epoch = (len(loader_train) + args.grad_accum_steps - 1
        ) // args.grad_accum_steps
    lr_scheduler, num_epochs = create_scheduler_v2(optimizer, **
        scheduler_kwargs(args), updates_per_epoch=updates_per_epoch)
    start_epoch = 0
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)
    if utils.is_primary(args):
        _logger.info(
            f"Scheduled epochs: {num_epochs}. LR stepped per {'epoch' if lr_scheduler.t_in_epochs else 'update'}."
            )
    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, 'set_epoch'):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, 'set_epoch'
                ):
                loader_train.sampler.set_epoch(epoch)
            train_metrics = train_one_epoch(epoch, model, loader_train,
                optimizer, train_loss_fn, args, lr_scheduler=lr_scheduler,
                saver=saver, output_dir=output_dir, amp_autocast=
                amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema,
                mixup_fn=mixup_fn, model_kd=model_kd)
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if utils.is_primary(args):
                    _logger.info(
                        'Distributing BatchNorm running means and vars')
                utils.distribute_bn(model, args.world_size, args.dist_bn ==
                    'reduce')
            eval_metrics = validate(model, loader_eval, validate_loss_fn,
                args, amp_autocast=amp_autocast)
            eval_metrics_unite = eval_metrics
            ema_eval_metrics = None
            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'
                    ):
                    utils.distribute_bn(model_ema, args.world_size, args.
                        dist_bn == 'reduce')
                ema_eval_metrics = validate(model_ema.module, loader_eval,
                    validate_loss_fn, args, amp_autocast=amp_autocast,
                    log_suffix=' (EMA)')
                if ema_eval_metrics[eval_metric] > eval_metrics[eval_metric]:
                    eval_metrics_unite = ema_eval_metrics
            if args.dryrun:
                break
            if output_dir is not None:
                lrs = [param_group['lr'] for param_group in optimizer.
                    param_groups]
                if args.log_tb:
                    for key, value in train_metrics.items():
                        writer.add_scalar('train/' + key, value, epoch)
                    for key, value in eval_metrics_unite.items():
                        writer.add_scalar('eval/' + key, value, epoch)
                    for i, lr in enumerate(lrs):
                        writer.add_scalar(f'lr/{i}', lr, epoch)
                utils.update_summary(epoch, train_metrics,
                    eval_metrics_unite, filename=os.path.join(output_dir,
                    'summary.csv'), lr=sum(lrs) / len(lrs), write_header=
                    best_metric is None, log_wandb=args.log_wandb and has_wandb
                    )
            if saver is not None:
                save_metric = eval_metrics[eval_metric]
                if ema_eval_metrics:
                    save_metric_ema = ema_eval_metrics[eval_metric]
                else:
                    save_metric_ema = -1
                best_metric, best_epoch = saver.save_checkpoint(epoch,
                    metric=save_metric, metric_ema=save_metric_ema)
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, eval_metrics_unite[eval_metric])
    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric,
            best_epoch))
