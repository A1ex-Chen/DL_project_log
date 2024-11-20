def build_test_loader(cfg):
    dataset_dicts = get_test_data_dicts(cfg.VISDRONE.TEST_JSON, cfg.
        VISDRONE.TEST_IMG_ROOT)
    dataset = DatasetFromList(dataset_dicts)
    mapper = Mapper(cfg, False)
    dataset = MapDataset(dataset, mapper)
    sampler = samplers.InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1,
        drop_last=False)
    data_loader = torch.utils.data.DataLoader(dataset, num_workers=cfg.
        DATALOADER.NUM_WORKERS, batch_sampler=batch_sampler, collate_fn=
        trivial_batch_collator)
    return data_loader
