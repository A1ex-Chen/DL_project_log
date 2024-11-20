def dataset_size(dataset):
    if isinstance(dataset, DaliDataLoader):
        return dataset.dataset_size
    else:
        return dataset.sampler.num_samples
