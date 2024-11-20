@smart_inference_mode()
def evaluate(model, dataloader, conf_thres=0.001, iou_thres=0.6, max_det=
    300, device='', single_cls=False, augment=False, half=True,
    compute_loss=None, num_classes=80, v8_eval=False):
    device = next(model.parameters()).device
    half &= device.type != 'cpu'
    model.half() if half else model.float()
    model.eval()
    cuda = device.type != 'cpu'
    nc = num_classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()
    seen = 0
    names = model.names if hasattr(model, 'names') else model.module.names
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R',
        'mAP50', 'mAP50-95')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = (0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0)
    dt = Profile(), Profile(), Profile()
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()
            im /= 255
            nb, _, height, width = im.shape
        with dt[1]:
            preds, train_out = model(im) if compute_loss else (model(im,
                augment=augment), None)
        if compute_loss:
            loss += compute_loss(train_out, targets)[1]
        targets[:, 2:] *= torch.tensor((width, height, width, height),
            device=device)
        lb = []
        with dt[2]:
            preds = non_max_suppression(preds, conf_thres, iou_thres,
                labels=lb, multi_label=True, agnostic=single_cls, max_det=
                max_det)
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            seen += 1
            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=
                        device), labels[:, 0]))
                continue
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = process_batch(predn, labelsn, iouv)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False,
            save_dir=None, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    model.float()
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return {'mAP@0.5': map50, 'mAP@0.5:0.95': map, 'P': mp, 'R': mr}
