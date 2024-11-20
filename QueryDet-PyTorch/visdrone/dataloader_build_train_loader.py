def build_train_loader(cfg):
    num_workers = get_world_size()
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    assert images_per_batch % num_workers == 0, 'SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).'.format(
        images_per_batch, num_workers)
    assert images_per_batch >= num_workers, 'SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).'.format(
        images_per_batch, num_workers)
    images_per_worker = images_per_batch // num_workers
    dataset_dicts = get_train_data_dicts(cfg.VISDRONE.TRAIN_JSON, cfg.
        VISDRONE.TRING_IMG_ROOT)
    dataset = DatasetFromList(dataset_dicts, copy=False)
    mapper = Mapper(cfg, True)
    dataset = MapDataset(dataset, mapper)
    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info('Using training sampler {}'.format(sampler_name))
    if sampler_name == 'TrainingSampler':
        sampler = samplers.TrainingSampler(len(dataset))
    elif sampler_name == 'RepeatFactorTrainingSampler':
        sampler = samplers.RepeatFactorTrainingSampler(dataset_dicts, cfg.
            DATALOADER.REPEAT_THRESHOLD)
    else:
        raise ValueError('Unknown training sampler: {}'.format(sampler_name))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler,
        images_per_worker, drop_last=True)
    data_loader = torch.utils.data.DataLoader(dataset, num_workers=cfg.
        DATALOADER.NUM_WORKERS, batch_sampler=batch_sampler, collate_fn=
        trivial_batch_collator, worker_init_fn=worker_init_reset_seed)
    return data_loader
