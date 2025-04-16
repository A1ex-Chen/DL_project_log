@smart_inference_mode()
def run(data, weights=None, batch_size=32, imgsz=640, conf_thres=0.001,
    iou_thres=0.6, max_det=300, task='val', device='', workers=8,
    single_cls=False, augment=False, verbose=False, save_txt=False,
    save_hybrid=False, save_conf=False, save_json=False, project=ROOT /
    'runs/val-seg', name='exp', exist_ok=False, half=True, dnn=False, model
    =None, dataloader=None, save_dir=Path(''), plots=True, overlap=False,
    mask_downsample_ratio=1, compute_loss=None, callbacks=Callbacks()):
    if save_json:
        check_requirements('pycocotools>=2.0.6')
        process = process_mask_native
    else:
        process = process_mask
    training = model is not None
    if training:
        device, pt, jit, engine = next(model.parameters()
            ).device, True, False, False
        half &= device.type != 'cpu'
        model.half() if half else model.float()
        nm = de_parallel(model).model[-1].nm
    else:
        device = select_device(device, batch_size=batch_size)
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
            exist_ok=True)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=
            data, fp16=half)
        stride, pt, jit, engine = (model.stride, model.pt, model.jit, model
            .engine)
        imgsz = check_img_size(imgsz, s=stride)
        half = model.fp16
        nm = de_parallel(model).model.model[-1].nm if isinstance(model,
            SegmentationModel) else 32
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1
                LOGGER.info(
                    f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models'
                    )
        data = check_dataset(data)
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(
        f'coco{os.sep}val2017.txt')
    nc = 1 if single_cls else int(data['nc'])
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()
    if not training:
        if pt and not single_cls:
            ncm = model.model.nc
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))
        pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)
        task = task if task in ('train', 'val', 'test') else 'val'
        dataloader = create_dataloader(data[task], imgsz, batch_size,
            stride, single_cls, pad=pad, rect=rect, workers=workers, prefix
            =colorstr(f'{task}: '), overlap_mask=overlap,
            mask_downsample_ratio=mask_downsample_ratio)[0]
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else model.module.names
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%22s' + '%11s' * 10) % ('Class', 'Images', 'Instances', 'Box(P',
        'R', 'mAP50', 'mAP50-95)', 'Mask(P', 'R', 'mAP50', 'mAP50-95)')
    dt = Profile(), Profile(), Profile()
    metrics = Metrics()
    loss = torch.zeros(4, device=device)
    jdict, stats = [], []
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)
    for batch_i, (im, targets, paths, shapes, masks) in enumerate(pbar):
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
                masks = masks.to(device)
            masks = masks.float()
            im = im.half() if half else im.float()
            im /= 255
            nb, _, height, width = im.shape
        with dt[1]:
            preds, protos, train_out = model(im) if compute_loss else (*
                model(im, augment=augment)[:2], None)
        if compute_loss:
            loss += compute_loss((train_out, protos), targets, masks)[1]
        targets[:, 2:] *= torch.tensor((width, height, width, height),
            device=device)
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)
            ] if save_hybrid else []
        with dt[2]:
            preds = non_max_suppression(preds, conf_thres, iou_thres,
                labels=lb, multi_label=True, agnostic=single_cls, max_det=
                max_det, nm=nm)
        plot_masks = []
        for si, (pred, proto) in enumerate(zip(preds, protos)):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]
            path, shape = Path(paths[si]), shapes[si][0]
            correct_masks = torch.zeros(npr, niou, dtype=torch.bool, device
                =device)
            correct_bboxes = torch.zeros(npr, niou, dtype=torch.bool,
                device=device)
            seen += 1
            if npr == 0:
                if nl:
                    stats.append((correct_masks, correct_bboxes, *torch.
                        zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None,
                            labels=labels[:, 0])
                continue
            midx = [si] if overlap else targets[:, 0] == si
            gt_masks = masks[midx]
            pred_masks = process(proto, pred[:, 6:], pred[:, :4], shape=im[
                si].shape[1:])
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct_bboxes = process_batch(predn, labelsn, iouv)
                correct_masks = process_batch(predn, labelsn, iouv,
                    pred_masks, gt_masks, overlap=overlap, masks=True)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct_masks, correct_bboxes, pred[:, 4], pred[:,
                5], labels[:, 0]))
            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            if plots and batch_i < 3:
                plot_masks.append(pred_masks[:15])
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir /
                    'labels' / f'{path.stem}.txt')
            if save_json:
                pred_masks = scale_image(im[si].shape[1:], pred_masks.
                    permute(1, 2, 0).contiguous().cpu().numpy(), shape,
                    shapes[si][1])
                save_one_json(predn, jdict, path, class_map, pred_masks)
        if plots and batch_i < 3:
            if len(plot_masks):
                plot_masks = torch.cat(plot_masks, dim=0)
            plot_images_and_masks(im, targets, masks, paths, save_dir /
                f'val_batch{batch_i}_labels.jpg', names)
            plot_images_and_masks(im, output_to_target(preds, max_det=15),
                plot_masks, paths, save_dir /
                f'val_batch{batch_i}_pred.jpg', names)
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    if len(stats) and stats[0].any():
        results = ap_per_class_box_and_mask(*stats, plot=plots, save_dir=
            save_dir, names=names)
        metrics.update(results)
    nt = np.bincount(stats[4].astype(int), minlength=nc)
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 8
    LOGGER.info(pf % ('all', seen, nt.sum(), *metrics.mean_results()))
    if nt.sum() == 0:
        LOGGER.warning(
            f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels'
            )
    if (verbose or nc < 50 and not training) and nc > 1 and len(stats):
        for i, c in enumerate(metrics.ap_class_index):
            LOGGER.info(pf % (names[c], seen, nt[c], *metrics.class_result(i)))
    t = tuple(x.t / seen * 1000.0 for x in dt)
    if not training:
        shape = batch_size, 3, imgsz, imgsz
        LOGGER.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}'
             % t)
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    (mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask,
        map_mask) = metrics.mean_results()
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights
            ).stem if weights is not None else ''
        anno_json = str(Path(
            '../datasets/coco/annotations/instances_val2017.json'))
        pred_json = str(save_dir / f'{w}_predictions.json')
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            anno = COCO(anno_json)
            pred = anno.loadRes(pred_json)
            results = []
            for eval in (COCOeval(anno, pred, 'bbox'), COCOeval(anno, pred,
                'segm')):
                if is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in
                        dataloader.dataset.im_files]
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                results.extend(eval.stats[:2])
            map_bbox, map50_bbox, map_mask, map50_mask = results
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')
    model.float()
    if not training:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
             if save_txt else '')
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    final_metric = (mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask,
        mr_mask, map50_mask, map_mask)
    return (*final_metric, *(loss.cpu() / len(dataloader)).tolist()
        ), metrics.get_maps(nc), t
