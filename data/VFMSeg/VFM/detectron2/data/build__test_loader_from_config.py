def _test_loader_from_config(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]
    dataset = get_detection_dataset_dicts(dataset_name, filter_empty=False,
        proposal_files=[cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.
        TEST).index(x)] for x in dataset_name] if cfg.MODEL.LOAD_PROPOSALS else
        None)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    return {'dataset': dataset, 'mapper': mapper, 'num_workers': cfg.
        DATALOADER.NUM_WORKERS, 'sampler': InferenceSampler(len(dataset))}
