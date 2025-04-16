def get_dataset_fn(data_path, dataset_type):
    if dataset_type == 'webdataset':
        return get_wds_dataset
    elif dataset_type == 'csv':
        return get_csv_dataset
    elif dataset_type == 'auto':
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f'Tried to figure out dataset type, but failed for extention {ext}.'
                )
    elif dataset_type == 'toy':
        return get_toy_dataset
    else:
        raise ValueError(f'Unsupported dataset type: {dataset_type}')
