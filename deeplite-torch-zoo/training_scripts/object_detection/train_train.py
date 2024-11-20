def train(opt, device):
    (hyp, save_dir, epochs, batch_size, single_cls, noval, nosave, workers,
        freeze) = (opt.hyp, Path(opt.save_dir), opt.epochs, opt.batch_size,
        opt.single_cls, opt.noval, opt.nosave, opt.workers, opt.freeze)
    w = save_dir / 'weights'
    w.mkdir(parents=True, exist_ok=True)
    last, best = w / 'last.pt', w / 'best.pt'
    if isinstance(hyp, (str, Path)):
        with open(hyp, errors='ignore') as f:
            hyp = Dict(yaml.safe_load(f))
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k,
        v in hyp.items()))
    opt.hyp = dict(hyp.copy())
    yaml_save(save_dir / 'hyp.yaml', hyp)
    yaml_save(save_dir / 'opt.yaml', vars(opt))
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    model = get_model(model_name=opt.model_name, dataset_name=opt.
        pretraining_dataset, pretrained=False).to(device)
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)
    dataloaders = get_dataloaders(data_root=opt.data_root, image_size=imgsz,
        dataset_name=opt.dataset, batch_size=batch_size // WORLD_SIZE,
        num_workers=workers, gs=gs, single_cls=single_cls, cache=opt.cache,
        rect=opt.rect, hsv_h=hyp.hsv_h, hsv_s=hyp.hsv_s, hsv_v=hyp.hsv_v,
        degrees=hyp.degrees, translate=hyp.translate, scale=hyp.scale,
        shear=hyp.shear, perspective=hyp.perspective, flipud=hyp.flipud,
        fliplr=hyp.fliplr, mosaic=hyp.mosaic, mixup=hyp.mixup, copy_paste=
        hyp.copy_paste)
    train_loader = dataloaders['train']
    val_loader = dataloaders['test']
    dataset = train_loader.dataset
    names = {(0): 'item'} if single_cls and len(dataset.data['names']
        ) != 1 else dataset.data['names']
    nc = 1 if single_cls else int(dataset.data['nc'])
    model = get_model(model_name=opt.model_name, dataset_name=opt.
        pretraining_dataset, pretrained=opt.pretrained, num_classes=nc).to(
        device)
    amp = not opt.no_amp
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(
        freeze[0]))]
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False
    nbs = 64
    accumulate = max(round(nbs / batch_size), 1)
    hyp['weight_decay'] *= batch_size * accumulate / nbs
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp[
        'momentum'], hyp['weight_decay'])
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    ema = ModelEMA(model) if RANK in {-1, 0} else None
    best_fitness, start_epoch = 0.0, 0
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            """WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.
See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started."""
            )
        model = torch.nn.DataParallel(model)
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')
    model.half().float()
    eval_function = get_eval_function(dataset_name=opt.dataset, model_name=
        opt.model_name)
    if cuda and RANK != -1:
        model = smart_DDP(model)
    nl = de_parallel(model).model[-1].nl
    hyp['box'] *= 3 / nl
    hyp['cls'] *= nc / 80 * 3 / nl
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc
    model.hyp = hyp
    model.names = names
    t0 = time.time()
    nb = len(train_loader)
    nw = max(round(hyp['warmup_epochs'] * nb), 100)
    last_opt_step = -1
    maps = np.zeros(nc)
    results = 0, 0, 0, 0, 0, 0, 0
    scheduler.last_epoch = start_epoch - 1
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = YOLOv5Loss(model)
    LOGGER.info(
        f"""Image sizes {imgsz} train, {imgsz} val
Using {train_loader.num_workers * WORLD_SIZE} dataloader workers
Logging results to {colorstr('bold', save_dir)}
Starting training for {epochs} epochs..."""
        )
    for epoch in range(start_epoch, epochs):
        model.train()
        mloss = torch.zeros(3, device=device)
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss',
            'obj_loss', 'cls_loss', 'Instances', 'Size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]
                    ).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j ==
                        0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp[
                            'warmup_momentum'], hyp['momentum']])
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs
                    ) // gs * gs
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [(math.ceil(x * sf / gs) * gs) for x in imgs.shape[2:]
                        ]
                    imgs = nn.functional.interpolate(imgs, size=ns, mode=
                        'bilinear', align_corners=False)
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(device))
                if RANK != -1:
                    loss *= WORLD_SIZE
                if opt.quad:
                    loss *= 4.0
            scaler.scale(loss).backward()
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm
                    =10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = (
                    f'{torch.cuda.memory_reserved() / 1000000000.0 if torch.cuda.is_available() else 0:.3g}G'
                    )
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0],
                    imgs.shape[-1]))
        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()
        if RANK in {-1, 0}:
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names',
                'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs or stopper.possible_stop
            if not noval or final_epoch:
                ap_dict = eval_function(ema.ema, val_loader, half=amp,
                    single_cls=single_cls, compute_loss=compute_loss)
            fi = ap_dict['mAP@0.5:0.95']
            stop = stopper(epoch=epoch, fitness=fi)
            if fi > best_fitness:
                best_fitness = fi
            if not opt.dryrun or not nosave or final_epoch:
                ckpt = {'epoch': epoch, 'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(), 'ema':
                    deepcopy(ema.ema).half(), 'updates': ema.updates,
                    'optimizer': optimizer.state_dict(), 'opt': vars(opt),
                    'date': datetime.now().isoformat()}
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
        if RANK != -1:
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)
            if RANK != 0:
                stop = broadcast_list[0]
        if stop or opt.dryrun:
            break
    if RANK in {-1, 0} and not opt.dryrun:
        LOGGER.info(
            f"""
{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours."""
            )
        for f in (last, best):
            if f.exists():
                strip_optimizer(f)
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    ckpt = torch.load(f, map_location=device)
                    model = ckpt['ema' if ckpt.get('ema') else 'model']
                    model.float().eval()
                    ap_dict = eval_function(model, val_loader, iou_thres=
                        0.65 if 'coco' in opt.dataset else 0.6, single_cls=
                        single_cls, compute_loss=compute_loss)
                    LOGGER.info(f'Final eval metrics: {ap_dict}')
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
    torch.cuda.empty_cache()
    return results
