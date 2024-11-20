def _train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    if dataset is None:
        dataset = get_detection_dataset_dicts(cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.
            MIN_KEYPOINTS_PER_IMAGE if cfg.MODEL.KEYPOINT_ON else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.
            LOAD_PROPOSALS else None)
        _log_api_usage('dataset.' + cfg.DATASETS.TRAIN[0])
    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        logger.info('Using training sampler {}'.format(sampler_name))
        if sampler_name == 'TrainingSampler':
            sampler = TrainingSampler(len(dataset))
        elif sampler_name == 'RepeatFactorTrainingSampler':
            repeat_factors = (RepeatFactorTrainingSampler.
                repeat_factors_from_category_frequency(dataset, cfg.
                DATALOADER.REPEAT_THRESHOLD))
            sampler = RepeatFactorTrainingSampler(repeat_factors)
        elif sampler_name == 'RandomSubsetTrainingSampler':
            sampler = RandomSubsetTrainingSampler(len(dataset), cfg.
                DATALOADER.RANDOM_SUBSET_RATIO)
        else:
            raise ValueError('Unknown training sampler: {}'.format(
                sampler_name))
    return {'dataset': dataset, 'sampler': sampler, 'mapper': mapper,
        'total_batch_size': cfg.SOLVER.IMS_PER_BATCH,
        'aspect_ratio_grouping': cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        'num_workers': cfg.DATALOADER.NUM_WORKERS}
