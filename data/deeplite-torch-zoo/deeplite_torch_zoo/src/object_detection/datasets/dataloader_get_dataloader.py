def get_dataloader(dataset_path, data, cfg, batch_size=16, gs=32, rank=0,
    mode='train'):
    with torch_distributed_zero_first(rank):
        dataset = build_yolo_dataset(cfg, dataset_path, batch_size, data,
            mode=mode, rect=mode == 'val', stride=gs)
    shuffle = mode == 'train'
    if getattr(dataset, 'rect', False) and shuffle:
        LOGGER.warning(
            "WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False"
            )
        shuffle = False
    workers = cfg.workers if mode == 'train' else cfg.workers * 2
    return build_dataloader(dataset, batch_size, workers, shuffle, rank)
