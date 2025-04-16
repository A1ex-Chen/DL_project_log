def main(args):
    utils.init_distributed_mode(args)
    global best_acc1
    if utils.is_main_process() and args.wandb:
        wandb_id = os.path.split(args.output_dir)[-1]
        wandb.init(project='ULIP', id=wandb_id, config=args, reinit=True,
            entity='lxue')
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.evaluate_3d:
        zero_stats = test_zeroshot_3d(args)
        print(zero_stats)
        return
    if args.use_scanrefer:
        ckpt = torch.load(args.test_ckpt_addr, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        model = getattr(models, args.model)(args=args)
        model.cuda()
        model.load_state_dict(state_dict, strict=False)
        print("=> loaded resume checkpoint '{}'".format(args.test_ckpt_addr))
    else:
        print('=> creating model: {}'.format(args.model))
        model = getattr(models, args.model)(args=args)
        model.cuda(args.gpu)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids
            =[args.gpu], bucket_cap_mb=200, find_unused_parameters=False)
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            print('in optimizer freeze {}'.format(n))
            continue
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
    optim_params = [{'params': p_wd, 'weight_decay': args.wd}, {'params':
        p_non_wd, 'weight_decay': 0}]
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.
        betas, eps=args.eps, weight_decay=args.wd)
    scaler = amp.GradScaler(enabled=not args.disable_amp)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            args.start_epoch = epoch
            result = model.load_state_dict(checkpoint['state_dict'], strict
                =False)
            print(result)
            optimizer.load_state_dict(checkpoint['optimizer']
                ) if 'optimizer' in checkpoint else ()
            scaler.load_state_dict(checkpoint['scaler']
                ) if 'scaler' in checkpoint else ()
            best_acc1 = checkpoint['best_acc1']
            print("=> loaded resume checkpoint '{}' (epoch {})".format(args
                .resume, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        latest = os.path.join(args.output_dir, 'checkpoint.pt')
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu')
            args.start_epoch = latest_checkpoint['epoch']
            model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            scaler.load_state_dict(latest_checkpoint['scaler'])
            best_acc1 = latest_checkpoint['best_acc1']
            print("=> loaded latest checkpoint '{}' (epoch {})".format(
                latest, latest_checkpoint['epoch']))
    cudnn.benchmark = True
    print('=> creating dataset')
    tokenizer = SimpleTokenizer()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,
        0.224, 0.225])
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224,
        scale=(0.5, 1.0)), transforms.ToTensor(), normalize])
    train_dataset = get_dataset(train_transform, tokenizer, args, 'train')
    train_dataset[0]
    val_dataset = get_dataset(None, tokenizer, args, 'val')
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset)
    else:
        train_sampler = None
        val_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=
        args.batch_size, shuffle=train_sampler is None, num_workers=args.
        workers, pin_memory=True, sampler=train_sampler, drop_last=True,
        collate_fn=customized_collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.
        batch_size, shuffle=val_sampler is None, num_workers=args.workers,
        pin_memory=True, sampler=val_sampler, drop_last=False)
    lr_schedule = utils.cosine_scheduler(args.lr, args.lr_end, args.epochs,
        len(train_loader) // args.update_freq, warmup_epochs=args.
        warmup_epochs, start_warmup_value=args.lr_start)
    criterion = models.get_loss(args).cuda(args.gpu)
    print(args)
    print('=> beginning training')
    best_epoch = -1
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        val_stats = {'acc1': -1}
        train_stats = train(train_loader, model, criterion, optimizer,
            scaler, epoch, lr_schedule, args)
        if epoch % 1 == 0:
            val_stats = test_zeroshot_3d_core(val_loader, model, tokenizer,
                args)
            acc1 = val_stats['acc1']
            print(val_stats)
            is_best = acc1 > best_acc1
            if is_best:
                best_epoch = epoch
            best_acc1 = max(acc1, best_acc1)
            if is_best or epoch % 50 == 0:
                print('=> saving checkpoint')
                utils.save_on_master({'epoch': epoch + 1, 'state_dict':
                    model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(), 'best_acc1': best_acc1,
                    'args': args}, is_best, args.output_dir)
            if epoch + 1 == args.epochs:
                print('=> saving last checkpoint')
                utils.save_on_master({'epoch': 'last', 'state_dict': model.
                    state_dict(), 'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(), 'best_acc1': best_acc1,
                    'args': args}, True, args.output_dir)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in val_stats.items()}, 'epoch':
            epoch, 'best_acc1': best_acc1, 'best_epoch': best_epoch}
        if utils.is_main_process():
            if args.wandb:
                wandb.log(log_stats)
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
