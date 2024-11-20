@deprecated
def get_data_splits_by_name(data_root, dataset_name, **kwargs):
    return get_dataloaders(data_root, dataset_name, **kwargs)
