def get_dataset_fn(dataset_type):
    if dataset_type == 'webdataset':
        return get_wds_dataset
    elif dataset_type == 'toy':
        return get_toy_dataset
    else:
        raise ValueError(f'Unsupported dataset type: {dataset_type}')
