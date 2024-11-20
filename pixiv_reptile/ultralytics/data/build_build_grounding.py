def build_grounding(cfg, img_path, json_file, batch, mode='train', rect=
    False, stride=32):
    """Build YOLO Dataset."""
    return GroundingDataset(img_path=img_path, json_file=json_file, imgsz=
        cfg.imgsz, batch_size=batch, augment=mode == 'train', hyp=cfg, rect
        =cfg.rect or rect, cache=cfg.cache or None, single_cls=cfg.
        single_cls or False, stride=int(stride), pad=0.0 if mode == 'train'
         else 0.5, prefix=colorstr(f'{mode}: '), task=cfg.task, classes=cfg
        .classes, fraction=cfg.fraction if mode == 'train' else 1.0)
