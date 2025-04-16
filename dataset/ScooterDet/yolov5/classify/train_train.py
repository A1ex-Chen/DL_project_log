def train(opt, device):
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    save_dir, data, bs, epochs, nw, imgsz, pretrained = opt.save_dir, Path(opt
        .data), opt.batch_size, opt.epochs, min(os.cpu_count() - 1, opt.workers
        ), opt.imgsz, str(opt.pretrained).lower() == 'true'
    cuda = device.type != 'cpu'
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    last, best = wdir / 'last.pt', wdir / 'best.pt'
    yaml_save(save_dir / 'opt.yaml', vars(opt))
    logger = GenericLogger(opt=opt, console_logger=LOGGER) if RANK in {-1, 0
        } else None
    with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
        data_dir = data if data.is_dir() else DATASETS_DIR / data
        if not data_dir.is_dir():
            LOGGER.info(
                f'\nDataset not found ⚠️, missing path {data_dir}, attempting download...'
                )
            t = time.time()
            if str(data) == 'imagenet':
                subprocess.run(['bash', str(ROOT /
                    'data/scripts/get_imagenet.sh')], shell=True, check=True)
            else:
                url = (
                    f'https://github.com/ultralytics/yolov5/releases/download/v1.0/{data}.zip'
                    )
                download(url, dir=data_dir.parent)
            s = f"""Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}
"""
            LOGGER.info(s)
    nc = len([x for x in (data_dir / 'train').glob('*') if x.is_dir()])
    trainloader = create_classification_dataloader(path=data_dir / 'train',
        imgsz=imgsz, batch_size=bs // WORLD_SIZE, augment=True, cache=opt.
        cache, rank=LOCAL_RANK, workers=nw)
    test_dir = data_dir / 'test' if (data_dir / 'test').exists(
        ) else data_dir / 'val'
    if RANK in {-1, 0}:
        testloader = create_classification_dataloader(path=test_dir, imgsz=
            imgsz, batch_size=bs // WORLD_SIZE * 2, augment=False, cache=
            opt.cache, rank=-1, workers=nw)
    with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
        if Path(opt.model).is_file() or opt.model.endswith('.pt'):
            model = attempt_load(opt.model, device='cpu', fuse=False)
        elif opt.model in torchvision.models.__dict__:
            model = torchvision.models.__dict__[opt.model](weights=
                'IMAGENET1K_V1' if pretrained else None)
        else:
            m = hub.list('ultralytics/yolov5')
            raise ModuleNotFoundError(
                f'--model {opt.model} not found. Available models are: \n' +
                '\n'.join(m))
        if isinstance(model, DetectionModel):
            LOGGER.warning(
                "WARNING ⚠️ pass YOLOv5 classifier model with '-cls' suffix, i.e. '--model yolov5s-cls.pt'"
                )
            model = ClassificationModel(model=model, nc=nc, cutoff=opt.
                cutoff or 10)
        reshape_classifier_output(model, nc)
    for m in model.modules():
        if not pretrained and hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        if isinstance(m, torch.nn.Dropout) and opt.dropout is not None:
            m.p = opt.dropout
    for p in model.parameters():
        p.requires_grad = True
    model = model.to(device)
    if RANK in {-1, 0}:
        model.names = trainloader.dataset.classes
        model.transforms = testloader.dataset.torch_transforms
        model_info(model)
        if opt.verbose:
            LOGGER.info(model)
        images, labels = next(iter(trainloader))
        file = imshow_cls(images[:25], labels[:25], names=model.names, f=
            save_dir / 'train_images.jpg')
        logger.log_images(file, name='Train Examples')
        logger.log_graph(model, imgsz)
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
    val = test_dir.stem
    LOGGER.info(
        f"""Image sizes {imgsz} train, {imgsz} test
Using {nw * WORLD_SIZE} dataloader workers
Logging results to {colorstr('bold', save_dir)}
Starting {opt.model} training on {data} dataset with {nc} classes for {epochs} epochs...

{'Epoch':>10}{'GPU_mem':>10}{'train_loss':>12}{f'{val}_loss':>12}{'top1_acc':>12}{'top5_acc':>12}"""
        )
    for epoch in range(epochs):
        tloss, vloss, fitness = 0.0, 0.0, 0.0
        model.train()
        if RANK != -1:
            trainloader.sampler.set_epoch(epoch)
        pbar = enumerate(trainloader)
        if RANK in {-1, 0}:
            pbar = tqdm(enumerate(trainloader), total=len(trainloader),
                bar_format=TQDM_BAR_FORMAT)
        for i, (images, labels) in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(
                device)
            with amp.autocast(enabled=cuda):
                loss = criterion(model(images), labels)
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
                if i == len(pbar) - 1:
                    top1, top5, vloss = validate.run(model=ema.ema,
                        dataloader=testloader, criterion=criterion, pbar=pbar)
                    fitness = top1
        scheduler.step()
        if RANK in {-1, 0}:
            if fitness > best_fitness:
                best_fitness = fitness
            metrics = {'train/loss': tloss, f'{val}/loss': vloss,
                'metrics/accuracy_top1': top1, 'metrics/accuracy_top5':
                top5, 'lr/0': optimizer.param_groups[0]['lr']}
            logger.log_metrics(metrics, epoch)
            final_epoch = epoch + 1 == epochs
            if not opt.nosave or final_epoch:
                ckpt = {'epoch': epoch, 'best_fitness': best_fitness,
                    'model': deepcopy(ema.ema).half(), 'ema': None,
                    'updates': ema.updates, 'optimizer': None, 'opt': vars(
                    opt), 'git': GIT_INFO, 'date': datetime.now().isoformat()}
                torch.save(ckpt, last)
                if best_fitness == fitness:
                    torch.save(ckpt, best)
                del ckpt
    if RANK in {-1, 0} and final_epoch:
        LOGGER.info(
            f"""
Training complete ({(time.time() - t0) / 3600:.3f} hours)
Results saved to {colorstr('bold', save_dir)}
Predict:         python classify/predict.py --weights {best} --source im.jpg
Validate:        python classify/val.py --weights {best} --data {data_dir}
Export:          python export.py --weights {best} --include onnx
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{best}')
Visualize:       https://netron.app
"""
            )
        images, labels = (x[:25] for x in next(iter(testloader)))
        pred = torch.max(ema.ema(images.to(device)), 1)[1]
        file = imshow_cls(images, labels, pred, de_parallel(model).names,
            verbose=False, f=save_dir / 'test_images.jpg')
        meta = {'epochs': epochs, 'top1_acc': best_fitness, 'date':
            datetime.now().isoformat()}
        logger.log_images(file, name='Test Examples (true-predicted)',
            epoch=epoch)
        logger.log_model(best, epochs, metadata=meta)
