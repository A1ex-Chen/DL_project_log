def train(opt, device):
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    save_dir, bs, epochs, nw, imgsz, pretrained = (opt.save_dir, opt.
        batch_size, opt.epochs, min(os.cpu_count() - 1, opt.workers), opt.
        imgsz, str(opt.pretrained).lower() == 'true')
    cuda = device.type != 'cpu'
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    last, best, best_sd = (wdir / 'last.pt', wdir / 'best.pt', wdir /
        'best_state_dict.pt')
    yaml_save(save_dir / 'opt.yaml', vars(opt))
    logger = GenericLogger(opt=opt, console_logger=LOGGER) if RANK in {-1, 0
        } else None
    dataloaders = get_dataloaders(data_root=opt.data_root, dataset_name=opt
        .dataset, batch_size=bs, test_batch_size=opt.test_batch_size,
        img_size=imgsz, num_workers=nw)
    trainloader, testloader = dataloaders['train'], dataloaders['test']
    opt.num_classes = len(trainloader.dataset.classes)
    with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
        model = get_model(model_name=opt.model, dataset_name=opt.
            pretraining_dataset, num_classes=opt.num_classes, pretrained=
            pretrained)
        model_kd = None
        if opt.kd_model_name is not None:
            model_kd = KDTeacher(opt)
    for p in model.parameters():
        p.requires_grad = True
    model = model.to(device)
    evaluation_fn = get_eval_function(model_name=opt.model, dataset_name=
        opt.pretraining_dataset)
    if RANK in {-1, 0}:
        if opt.verbose:
            LOGGER.info(model)
    optimizer = smart_optimizer(model, opt.optimizer, opt.lr0, momentum=0.9,
        decay=opt.decay)
    lrf = 0.01
    lf = lambda x: (1 - x / epochs) * (1 - lrf) + lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    ema = ModelEMA(model) if RANK in {-1, 0} else None
    if cuda and RANK != -1:
        model = smart_DDP(model)
    t0 = time.time()
    criterion = smartCrossEntropyLoss(label_smoothing=opt.label_smoothing)
    best_fitness = 0.0
    scaler = amp.GradScaler(enabled=cuda)
    LOGGER.info(
        f"""Image sizes {imgsz} train, {imgsz} test
Using {nw * WORLD_SIZE} dataloader workers
Logging results to {colorstr('bold', save_dir)}
Starting {opt.model} training on {opt.dataset} dataset for {epochs} epochs...

{'Epoch':>10}{'GPU_mem':>10}{'train_loss':>12}{'top1_acc':>12}{'top5_acc':>12}"""
        )
    for epoch in range(epochs):
        tloss, vloss, fitness = 0.0, 0.0, 0.0
        model.train()
        pbar = enumerate(trainloader)
        if RANK in {-1, 0}:
            pbar = tqdm(enumerate(trainloader), total=len(trainloader),
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (images, labels) in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(
                device)
            with amp.autocast(enabled=cuda):
                output = model(images)
                loss = criterion(output, labels)
                if model_kd is not None:
                    kd_loss = compute_kd_loss(images, output, model_kd, model)
                    if not opt.use_kd_loss_only:
                        loss += opt.alpha_kd * kd_loss
                    else:
                        loss = opt.alpha_kd * kd_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)
            if RANK in {-1, 0}:
                tloss = (tloss * i + loss.item()) / (i + 1)
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 
                    1000000000.0 if torch.cuda.is_available() else 0)
                pbar.desc = (
                    f"{f'{epoch + 1}/{epochs}':>10}{mem:>10}{tloss:>12.3g}" +
                    ' ' * 36)
                if opt.dryrun or i == len(pbar) - 1:
                    metrics = evaluation_fn(ema.ema, testloader,
                        progressbar=False, break_iter=None if not opt.
                        dryrun else 1)
                    top1, top5 = metrics['acc'], metrics['acc_top5']
                    fitness = top1
                    pbar.desc = f'{pbar.desc[:-36]}{top1:>12.3g}{top5:>12.3g}'
            if opt.dryrun:
                break
        scheduler.step()
        if opt.dryrun:
            break
        if RANK in {-1, 0}:
            if fitness > best_fitness:
                best_fitness = fitness
            metrics = {'train/loss': tloss, 'metrics/accuracy_top1': top1,
                'metrics/accuracy_top5': top5, 'lr/0': optimizer.
                param_groups[0]['lr']}
            logger.log_metrics(metrics, epoch)
            final_epoch = epoch + 1 == epochs
            if not opt.nosave or final_epoch:
                ckpt = {'epoch': epoch, 'best_fitness': best_fitness,
                    'model': deepcopy(ema.ema).half(), 'ema': None,
                    'updates': ema.updates, 'optimizer': None, 'opt': vars(
                    opt), 'date': datetime.now().isoformat()}
                torch.save(ckpt, last)
                if best_fitness == fitness:
                    torch.save(ckpt, best)
                    torch.save(ckpt['model'].state_dict(), best_sd)
                del ckpt
    if not opt.dryrun and RANK in {-1, 0} and final_epoch:
        LOGGER.info(
            f"""
Training complete ({(time.time() - t0) / 3600:.3f} hours)
Results saved to {colorstr('bold', save_dir)}"""
            )
        meta = {'epochs': epochs, 'top1_acc': best_fitness, 'date':
            datetime.now().isoformat()}
        logger.log_model(best, epochs, metadata=meta)
