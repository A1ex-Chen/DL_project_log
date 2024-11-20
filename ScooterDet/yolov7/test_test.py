def test(data, weights=None, batch_size=32, imgsz=640, conf_thres=0.001,
    iou_thres=0.6, save_json=False, single_cls=False, augment=False,
    verbose=False, model=None, dataloader=None, save_dir=Path(''), save_txt
    =False, save_hybrid=False, save_conf=False, plots=True, wandb_logger=
    None, compute_loss=None, half_precision=True, trace=False, is_coco=
    False, v5_metric=False):
    training = model is not None
    if training:
        device = next(model.parameters()).device
    else:
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)
        save_dir = Path(increment_path(Path(opt.project) / opt.name,
            exist_ok=opt.exist_ok))
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
            exist_ok=True)
        model = attempt_load(weights, map_location=device)
        gs = max(int(model.stride.max()), 32)
        imgsz = check_img_size(imgsz, s=gs)
        if trace:
            model = TracedModel(model, device, imgsz)
    half = device.type != 'cpu' and half_precision
    if half:
        model.half()
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)
    nc = 1 if single_cls else int(data['nc'])
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(
                model.parameters())))
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs,
            opt, pad=0.5, rect=True, prefix=colorstr(f'{task}: '))[0]
    if v5_metric:
        print('Testing with YOLOv5 AP metric...')
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model,
        'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R',
        'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0)
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader,
        desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()
        img /= 255.0
        targets = targets.to(device)
        nb, _, height, width = img.shape
        with torch.no_grad():
            t = time_synchronized()
            out, train_out = model(img, augment=augment)
            t0 += time_synchronized() - t
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1
                    ][:3]
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(
                device)
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)
                ] if save_hybrid else []
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres
                =iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            path = Path(paths[si])
            seen += 1
            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool),
                        torch.Tensor(), torch.Tensor(), tcls))
                continue
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0],
                shapes[si][1])
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn
                        ).view(-1).tolist()
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a'
                        ) as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:
                if (wandb_logger.current_epoch % wandb_logger.bbox_interval ==
                    0):
                    box_data = [{'position': {'minX': xyxy[0], 'minY': xyxy
                        [1], 'maxX': xyxy[2], 'maxY': xyxy[3]}, 'class_id':
                        int(cls), 'box_caption': '%s %.3f' % (names[cls],
                        conf), 'scores': {'class_score': conf}, 'domain':
                        'pixel'} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {'predictions': {'box_data': box_data,
                        'class_labels': names}}
                    wandb_images.append(wandb_logger.wandb.Image(img[si],
                        boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names
                ) if wandb_logger and wandb_logger.wandb_run else None
            if save_json:
                image_id = int(path.stem) if path.stem.isnumeric(
                    ) else path.stem
                box = xyxy2xywh(predn[:, :4])
                box[:, :2] -= box[:, 2:] / 2
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id, 'category_id': 
                        coco91class[int(p[5])] if is_coco else int(p[5]),
                        'bbox': [round(x, 3) for x in b], 'score': round(p[
                        4], 5)})
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool,
                device=device)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes
                    [si][1])
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels
                        [:, 0:1], tbox), 1))
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)
                    if pi.shape[0]:
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv
                                if len(detected) == nl:
                                    break
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(),
                tcls))
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'
            Thread(target=plot_images, args=(img, targets, paths, f, names),
                daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'
            Thread(target=plot_images, args=(img, output_to_target(out),
                paths, f, names), daemon=True).start()
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric
            =v5_metric, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
    else:
        nt = torch.zeros(1)
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if (verbose or nc < 50 and not training) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    t = tuple(x / seen * 1000.0 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz,
        batch_size)
    if not training:
        print(
            'Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g'
             % t)
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for
                f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({'Validation': val_batches})
    if wandb_images:
        wandb_logger.log({'Bounding Box Debugger/Images': wandb_images})
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights
            ).stem if weights is not None else ''
        anno_json = './coco/annotations/instances_val2017.json'
        pred_json = str(save_dir / f'{w}_predictions.json')
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            anno = COCO(anno_json)
            pred = anno.loadRes(pred_json)
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader
                    .dataset.img_files]
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]
        except Exception as e:
            print(f'pycocotools unable to run: {e}')
    model.float()
    if not training:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
             if save_txt else '')
        print(f'Results saved to {save_dir}{s}')
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()
        ), maps, t
