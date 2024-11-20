def train(hyp, opt, device, callbacks):
    (save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg,
        resume, noval, nosave, workers, freeze) = (Path(opt.save_dir), opt.
        epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve,
        opt.data, opt.cfg, opt.resume, opt.noval, opt.nosave, opt.workers,
        opt.freeze)
    callbacks.run('on_pretrain_routine_start')
    w = save_dir / 'weights'
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)
    last, best = w / 'last.pt', w / 'best.pt'
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k,
        v in hyp.items()))
    opt.hyp = hyp.copy()
    if not evolve:
        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.safe_dump(hyp, f, sort_keys=False)
        with open(save_dir / 'opt.yaml', 'w') as f:
            yaml.safe_dump(vars(opt), f, sort_keys=False)
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)
        if loggers.clearml:
            data_dict = loggers.clearml.data_dict
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp, batch_size = (opt.weights, opt.epochs,
                    opt.hyp, opt.batch_size)
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))
    plots = not evolve and not opt.noplots
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])
    names = ['item'] if single_cls and len(data_dict['names']
        ) != 1 else data_dict['names']
    assert len(names
        ) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'
    is_coco = isinstance(val_path, str) and val_path.endswith(
        'coco/val2017.txt')
    check_suffix(weights, '.pt')
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)
        ckpt = torch.load(weights, map_location='cpu')
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.
            get('anchors')).to(device)
        exclude = ['anchor'] if (cfg or hyp.get('anchors')
            ) and not resume else []
        csd = ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
        model.load_state_dict(csd, strict=False)
        LOGGER.info(
            f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}'
            )
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
    amp = check_amp(model)
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(
        freeze[0]))]
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)
    if RANK == -1 and batch_size == -1:
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({'batch_size': batch_size})
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
    if pretrained:
        best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer,
            ema, weights, epochs, resume)
        del ckpt, csd
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            """WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.
See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started."""
            )
        model = torch.nn.DataParallel(model)
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')
    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size //
        WORLD_SIZE, gs, single_cls, hyp=hyp, augment=True, cache=None if 
        opt.cache == 'val' else opt.cache, rect=opt.rect, rank=LOCAL_RANK,
        workers=workers, image_weights=opt.image_weights, quad=opt.quad,
        prefix=colorstr('train: '), shuffle=True)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'
    if RANK in {-1, 0}:
        val_loader = create_dataloader(val_path, imgsz, batch_size //
            WORLD_SIZE * 2, gs, single_cls, hyp=hyp, cache=None if noval else
            opt.cache, rect=True, rank=-1, workers=workers * 2, pad=0.5,
            prefix=colorstr('val: '))[0]
        if not resume:
            if plots:
                plot_labels(labels, names, save_dir)
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'],
                    imgsz=imgsz)
            model.half().float()
        callbacks.run('on_pretrain_routine_end')
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    nl = de_parallel(model).model[-1].nl
    hyp['box'] *= 3 / nl
    hyp['cls'] *= nc / 80 * 3 / nl
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc
    model.hyp = hyp
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device
        ) * nc
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
    compute_loss = ComputeLoss(model)
    callbacks.run('on_train_start')
    LOGGER.info(
        f"""Image sizes {imgsz} train, {imgsz} val
Using {train_loader.num_workers * WORLD_SIZE} dataloader workers
Logging results to {colorstr('bold', save_dir)}
Starting training for {epochs} epochs..."""
        )
    for epoch in range(start_epoch, epochs):
        callbacks.run('on_train_epoch_start')
        model.train()
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
            iw = labels_to_image_weights(dataset.labels, nc=nc,
                class_weights=cw)
            dataset.indices = random.choices(range(dataset.n), weights=iw,
                k=dataset.n)
        mloss = torch.zeros(3, device=device)
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj',
            'cls', 'labels', 'img_size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=
                '{l_bar}{bar:10}{r_bar}{bar:-10b}')
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:
            callbacks.run('on_train_batch_start')
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
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
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
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0],
                    imgs.shape[-1]))
                callbacks.run('on_train_batch_end', ni, model, imgs,
                    targets, paths, plots)
                if callbacks.stop_training:
                    return
        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()
        if RANK in {-1, 0}:
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names',
                'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs or stopper.possible_stop
            if not noval or final_epoch:
                results, maps, _ = val.run(data_dict, batch_size=batch_size //
                    WORLD_SIZE * 2, imgsz=imgsz, half=amp, model=ema.ema,
                    single_cls=single_cls, dataloader=val_loader, save_dir=
                    save_dir, plots=False, callbacks=callbacks,
                    compute_loss=compute_loss)
            fi = fitness(np.array(results).reshape(1, -1))
            stop = stopper(epoch=epoch, fitness=fi)
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi
                )
            if not nosave or final_epoch and not evolve:
                ckpt = {'epoch': epoch, 'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(), 'ema':
                    deepcopy(ema.ema).half(), 'updates': ema.updates,
                    'optimizer': optimizer.state_dict(), 'wandb_id': 
                    loggers.wandb.wandb_run.id if loggers.wandb else None,
                    'opt': vars(opt), 'date': datetime.now().isoformat()}
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch,
                    best_fitness, fi)
        if RANK != -1:
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break
    if RANK in {-1, 0}:
        LOGGER.info(
            f"""
{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours."""
            )
        for f in (last, best):
            if f.exists():
                strip_optimizer(f)
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = val.run(data_dict, batch_size=
                        batch_size // WORLD_SIZE * 2, imgsz=imgsz, model=
                        attempt_load(f, device).half(), iou_thres=0.65 if
                        is_coco else 0.6, single_cls=single_cls, dataloader
                        =val_loader, save_dir=save_dir, save_json=is_coco,
                        verbose=True, plots=plots, callbacks=callbacks,
                        compute_loss=compute_loss)
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) +
                            list(results) + lr, epoch, best_fitness, fi)
        callbacks.run('on_train_end', last, best, plots, epoch, results)
    torch.cuda.empty_cache()
    return results
