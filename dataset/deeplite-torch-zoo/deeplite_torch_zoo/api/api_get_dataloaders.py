def get_dataloaders(data_root, dataset_name, **kwargs):
    """
    The datasets function calls in the format of (get_`dataset_name`_for_`model_name`).
    Except for classification since the datasets format for classification models is the same.
    The function calls for classification models are in the format (get_`dataset_name`)

    returns datasplits in the following format:
    {
       'train': train_data_loader,
       'test' : test_data_loader
    }
    """
    data_split_wrapper_fn = DATASET_WRAPPER_REGISTRY.get(dataset_name=
        dataset_name)
    data_split = data_split_wrapper_fn(data_root=data_root, **kwargs)
    return data_split
