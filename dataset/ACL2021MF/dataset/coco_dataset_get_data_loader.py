def get_data_loader(config, dataset):
    collate_fn = lambda d: data_wrapper(config, d)
    return DataLoader(dataset, batch_size=config.batch_size, num_workers=0,
        collate_fn=collate_fn)
