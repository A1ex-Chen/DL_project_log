def build_yolo_dataset(cfg, img_path, batch, data, mode='train', rect=False,
    stride=32):
    """Build YOLO Dataset"""
    return YOLODataset(img_path=img_path, imgsz=cfg.imgsz, batch_size=batch,
        augment=mode == 'train', hyp=cfg, rect=cfg.rect or rect, cache=cfg.
        cache or None, single_cls=cfg.single_cls or False, stride=int(
        stride), pad=0.0 if mode == 'train' else 0.5, prefix=colorstr(
        f'{mode}: '), use_segments=cfg.task == 'segment', use_keypoints=cfg
        .task == 'pose', classes=cfg.classes, data=data, fraction=cfg.
        fraction if mode == 'train' else 1.0)
