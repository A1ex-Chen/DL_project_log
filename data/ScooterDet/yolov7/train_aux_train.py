def train(hyp, opt, device, tb_writer=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k,
        v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = (Path(
        opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size,
        opt.weights, opt.global_rank)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    plots = not opt.evolve
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    is_coco = opt.data.endswith('coco.yaml')
    loggers = {'wandb': None}
    if rank in [-1, 0]:
        opt.hyp = hyp
        run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt'
            ) and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id,
            data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp
    nc = 1 if opt.single_cls else int(data_dict['nc'])
    names = ['item'] if opt.single_cls and len(data_dict['names']
        ) != 1 else data_dict['names']
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len
        (names), nc, opt.data)
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)
        ckpt = torch.load(weights, map_location=device)
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=
            hyp.get('anchors')).to(device)
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')
            ) and not opt.resume else []
        state_dict = ckpt['model'].float().state_dict()
        state_dict = intersect_dicts(state_dict, model.state_dict(),
            exclude=exclude)
        model.load_state_dict(state_dict, strict=False)
        logger.info('Transferred %g/%g items from %s' % (len(state_dict),
            len(model.state_dict()), weights))
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(
            device)
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)
    train_path = data_dict['train']
    test_path = data_dict['val']
    freeze = []
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False
    nbs = 64
    accumulate = max(round(nbs / total_batch_size), 1)
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'):
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'):
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):
                pg0.append(v.rbr_dense.vector)
    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 
            0.999))
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'],
            nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp[
        'weight_decay']})
    optimizer.add_param_group({'params': pg2})
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (
        len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    ema = ModelEMA(model) if rank in [-1, 0] else None
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (
                weights, epochs)
        if epochs < start_epoch:
            logger.info(
                '%s has been trained for %g epochs. Fine-tuning for %g additional epochs.'
                 % (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']
        del ckpt, state_dict
    gs = max(int(model.stride.max()), 32)
    nl = model.model[-1].nl
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size,
        gs, opt, hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.
        rect, rank=rank, world_size=opt.world_size, workers=opt.workers,
        image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr(
        'train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()
    nb = len(dataloader)
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (
        mlc, nc, opt.data, nc - 1)
    if rank in [-1, 0]:
        testloader = create_dataloader(test_path, imgsz_test, batch_size * 
            2, gs, opt, hyp=hyp, cache=opt.cache_images and not opt.notest,
            rect=True, rank=-1, world_size=opt.world_size, workers=opt.
            workers, pad=0.5, prefix=colorstr('val: '))[0]
        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])
            if plots:
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'],
                    imgsz=imgsz)
            model.half().float()
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.
            local_rank, find_unused_parameters=any(isinstance(layer, nn.
            MultiheadAttention) for layer in model.modules()))
    hyp['box'] *= 3.0 / nl
    hyp['cls'] *= nc / 80.0 * 3.0 / nl
    hyp['obj'] *= (imgsz / 640) ** 2 * 3.0 / nl
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc
    model.hyp = hyp
    model.gr = 1.0
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device
        ) * nc
    model.names = names
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)
    maps = np.zeros(nc)
    results = 0, 0, 0, 0, 0, 0, 0
    scheduler.last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss_ota = ComputeLossAuxOTA(model)
    compute_loss = ComputeLoss(model)
    logger.info(
        f"""Image sizes {imgsz} train, {imgsz_test} test
Using {dataloader.num_workers} dataloader workers
Logging results to {save_dir}
Starting training for {epochs} epochs..."""
        )
    torch.save(model, wdir / 'init.pt')
    for epoch in range(start_epoch, epochs):
        model.train()
        if opt.image_weights:
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
                iw = labels_to_image_weights(dataset.labels, nc=nc,
                    class_weights=cw)
                dataset.indices = random.choices(range(dataset.n), weights=
                    iw, k=dataset.n)
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else
                    torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()
        mloss = torch.zeros(4, device=device)
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj',
            'cls', 'total', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs /
                    total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j ==
                        2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp[
                            'warmup_momentum'], hyp['momentum']])
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [(math.ceil(x * sf / gs) * gs) for x in imgs.shape[2:]
                        ]
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear',
                        align_corners=False)
            with amp.autocast(enabled=cuda):
                pred = model(imgs)
                loss, loss_items = compute_loss_ota(pred, targets.to(device
                    ), imgs)
                if rank != -1:
                    loss *= opt.world_size
                if opt.quad:
                    loss *= 4.0
            scaler.scale(loss).backward()
            if ni % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 
                    1000000000.0 if torch.cuda.is_available() else 0)
                s = ('%10s' * 2 + '%10.4g' * 6) % ('%g/%g' % (epoch, epochs -
                    1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)
                if plots and ni < 10:
                    f = save_dir / f'train_batch{ni}.jpg'
                    Thread(target=plot_images, args=(imgs, targets, paths,
                        f), daemon=True).start()
                elif plots and ni == 10 and wandb_logger.wandb:
                    wandb_logger.log({'Mosaics': [wandb_logger.wandb.Image(
                        str(x), caption=x.name) for x in save_dir.glob(
                        'train*.jpg') if x.exists()]})
        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()
        if rank in [-1, 0]:
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr',
                'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:
                wandb_logger.current_epoch = epoch + 1
                results, maps, times = test.test(data_dict, batch_size=
                    batch_size * 2, imgsz=imgsz_test, model=ema.ema,
                    single_cls=opt.single_cls, dataloader=testloader,
                    save_dir=save_dir, verbose=nc < 50 and final_epoch,
                    plots=plots and final_epoch, wandb_logger=wandb_logger,
                    compute_loss=compute_loss, is_coco=is_coco, v5_metric=
                    opt.v5_metric)
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (
                    results_file, opt.bucket, opt.name))
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',
                'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5',
                'metrics/mAP_0.5:0.95', 'val/box_loss', 'val/obj_loss',
                'val/cls_loss', 'x/lr0', 'x/lr1', 'x/lr2']
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})
            fi = fitness(np.array(results).reshape(1, -1))
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)
            if not opt.nosave or final_epoch and not opt.evolve:
                ckpt = {'epoch': epoch, 'best_fitness': best_fitness,
                    'training_results': results_file.read_text(), 'model':
                    deepcopy(model.module if is_parallel(model) else model)
                    .half(), 'ema': deepcopy(ema.ema).half(), 'updates':
                    ema.updates, 'optimizer': optimizer.state_dict(),
                    'wandb_id': wandb_logger.wandb_run.id if wandb_logger.
                    wandb else None}
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if best_fitness == fi and epoch >= 200:
                    torch.save(ckpt, wdir / 'best_{:03d}.pt'.format(epoch))
                if epoch == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                elif (epoch + 1) % 25 == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                elif epoch >= epochs - 5:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch
                        ) and opt.save_period != -1:
                        wandb_logger.log_model(last.parent, opt, epoch, fi,
                            best_model=best_fitness == fi)
                del ckpt
    if rank in [-1, 0]:
        if plots:
            plot_results(save_dir=save_dir)
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[
                    f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({'Results': [wandb_logger.wandb.Image(str(
                    save_dir / f), caption=f) for f in files if (save_dir /
                    f).exists()]})
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch -
            start_epoch + 1, (time.time() - t0) / 3600))
        if opt.data.endswith('coco.yaml') and nc == 80:
            for m in ((last, best) if best.exists() else last):
                results, _, _ = test.test(opt.data, batch_size=batch_size *
                    2, imgsz=imgsz_test, conf_thres=0.001, iou_thres=0.7,
                    model=attempt_load(m, device).half(), single_cls=opt.
                    single_cls, dataloader=testloader, save_dir=save_dir,
                    save_json=True, plots=False, is_coco=is_coco, v5_metric
                    =opt.v5_metric)
        final = best if best.exists() else last
        for f in (last, best):
            if f.exists():
                strip_optimizer(f)
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')
        if wandb_logger.wandb and not opt.evolve:
            wandb_logger.wandb.log_artifact(str(final), type='model', name=
                'run_' + wandb_logger.wandb_run.id + '_model', aliases=[
                'last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results
