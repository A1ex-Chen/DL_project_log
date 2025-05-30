def build_dataloader(cfg, args, mode='train', domain='source',
    start_iteration=0, different_batch_size=None):
    assert mode in ['train', 'val', 'test', 'train_labeled', 'train_unlabeled']
    dataset_cfg = cfg.get('DATASET_' + domain.upper())
    split = dataset_cfg[mode.upper()]
    is_train = 'train' in mode
    batch_size = cfg['TRAIN'].BATCH_SIZE if is_train else cfg['VAL'].BATCH_SIZE
    if different_batch_size is not None:
        batch_size = different_batch_size
    dataset_kwargs = CN(dataset_cfg.get(dataset_cfg.TYPE, dict()))
    if 'SCN' in cfg.MODEL_3D.keys():
        assert dataset_kwargs.full_scale == cfg.MODEL_3D.SCN.full_scale
    augmentation = dataset_kwargs.pop('augmentation')
    augmentation = augmentation if is_train else dict()
    if domain == 'target' and (not is_train or mode == 'train_labeled'):
        dataset_kwargs.pop('pselab_paths')
    if dataset_cfg.TYPE == 'NuScenesLidarSegSCN':
        dataset = NuScenesLidarSegSCN(args=args, split=split, output_orig=
            not is_train, **dataset_kwargs, **augmentation)
    elif dataset_cfg.TYPE == 'A2D2SCN':
        dataset = A2D2SCN(args=args, is_train=is_train, split=split, **
            dataset_kwargs, **augmentation)
    elif dataset_cfg.TYPE == 'SemanticKITTISCN':
        dataset = SemanticKITTISCN(args=args, is_train=is_train, split=
            split, output_orig=not is_train, **dataset_kwargs, **augmentation)
    elif dataset_cfg.TYPE == 'VirtualKITTISCN':
        dataset = VirtualKITTISCN(args=args, is_train=is_train, split=split,
            **dataset_kwargs, **augmentation)
    else:
        raise ValueError('Unsupported type of dataset: {}.'.format(
            dataset_cfg.TYPE))
    if 'SCN' in dataset_cfg.TYPE:
        collate_fn = get_collate_scn(is_train, args.vfmlab)
    else:
        collate_fn = default_collate
    if is_train:
        sampler = RandomSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size,
            drop_last=cfg.DATALOADER.DROP_LAST)
        batch_sampler = IterationBasedBatchSampler(batch_sampler, cfg.
            SCHEDULER.MAX_ITERATION, start_iteration)
        dataloader = DataLoader(dataset, pin_memory=True, batch_sampler=
            batch_sampler, num_workers=cfg.DATALOADER.NUM_WORKERS,
            worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    else:
        dataloader = DataLoader(dataset, pin_memory=True, batch_size=
            batch_size, drop_last=False, num_workers=cfg.DATALOADER.
            NUM_WORKERS, worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    return dataloader
