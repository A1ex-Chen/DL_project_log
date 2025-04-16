def lp_main(args, device, writer, pretrain_epoch, idx):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    args.class_index_dict = load_class_label(args.class_label_path)
    clap_model, clap_model_cfg = create_model(args.amodel, args.tmodel,
        args.pretrained, precision=args.precision, device=device, jit=args.
        torchscript, force_quick_gelu=args.force_quick_gelu,
        openai_model_cache_dir=os.path.expanduser(args.
        openai_model_cache_dir), skip_params=False, enable_fusion=args.
        enable_fusion, fusion_type=args.fusion_type)
    args.lp_out_ch = len(list(args.class_index_dict.keys()))
    if idx == 0:
        logging.info(f'linear probe using mlp: {args.lp_mlp}')
        logging.info(f'linear probe using freeze: {args.lp_freeze}')
        logging.info(f'linear probe act layer: {args.lp_act}')
        logging.info(f'linear probe out ch: {args.lp_out_ch}')
        logging.info(
            f'linear probe learning rate (if applicable): {args.lp_lr}')
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
    if is_master(args) and idx == 0:
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
    if args.wandb and is_master(args):
        args.train_sz = data['train'].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data['val'].dataloader.num_samples
        if args.debug:
            wandb.watch(model, log='all')
        if idx == 0:
            wandb.save(params_file)
    best_metrics = {}
    if 'train' not in data:
        metric = evaluate(model, data, start_epoch, args, writer,
            extra_suffix='_pe@' + str(pretrain_epoch))
        if is_master(args):
            best_metrics = update_metric(best_metrics, metric)
        return
    elif start_epoch == 0 and 'val' in data and not args.no_eval:
        metric = evaluate(model, data, 0, args, writer, extra_suffix='_pe@' +
            str(pretrain_epoch))
        if is_master(args):
            best_metrics = update_metric(best_metrics, metric)
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
            args, writer, extra_suffix='_pe@' + str(pretrain_epoch))
        completed_epoch = epoch + 1
        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')
            ) and not args.no_eval:
            metric = evaluate(model, data, completed_epoch, args, writer,
                extra_suffix='_pe@' + str(pretrain_epoch))
            if is_master(args):
                best_metrics = update_metric(best_metrics, metric)
            if args.save_top_performance:
                top_k_dataset = args.top_k_checkpoint_select_dataset
                top_k_metric = args.top_k_checkpoint_select_metric
                filtered_metrics = [v for k, v in metric.items() if 
                    top_k_metric in k and top_k_dataset in k]
        if args.save_logs:
            opt_dict = {(k + '_' + 'optimizer'): v.state_dict() for k, v in
                optimizer.items()}
            checkpoint_dict = {'epoch': completed_epoch, 'pretrain_epoch':
                pretrain_epoch, 'name': args.name, 'state_dict': model.
                state_dict()}
            checkpoint_dict.update(opt_dict)
            if scaler is not None:
                checkpoint_dict['scaler'] = scaler.state_dict()
            if (completed_epoch == args.epochs or args.save_frequency > 0 and
                completed_epoch % args.save_frequency == 0):
                torch.save(checkpoint_dict, os.path.join(args.
                    checkpoint_path,
                    f'pretrain_epoch_{pretrain_epoch}_lp_epoch_{completed_epoch}.pt'
                    ))
            if args.save_most_recent:
                torch.save(checkpoint_dict, os.path.join(args.
                    checkpoint_path,
                    f'pretrain_epoch_{pretrain_epoch}_lp_epoch_latest.pt'))
            if args.save_top_performance and not args.no_eval:
                update_top_k_performance(filtered_metrics,
                    current_top_k_ckpt_metrics, args, checkpoint_dict,
                    bignumbetter=True, pretrain_epoch=pretrain_epoch)
    del clap_model
    return best_metrics
