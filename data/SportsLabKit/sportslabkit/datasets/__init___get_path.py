def get_path(dataset: (str | None)=None) ->str:
    """Get the path to the data file.

    Args:
        dataset (str): Name of the dataset. If None, print the available datasets.

    Returns:
        str: Path to the data file.
    """
    if dataset is None:
        print('Available keys:')
        for d in available:
            print(f' - {d}')
        return
    if dataset in _available_dir:
        return _available_dir[dataset]
    if dataset.split('/')[0] in _available_dir:
        ret_path = _available_dir[dataset.split('/')[0]] / dataset.split('/')[1
            ]
        assert ret_path.exists(), f'File {ret_path} not available'
        return ret_path
    msg = f"The dataset '{dataset}' is not available. "
    msg += f"Available datasets are {', '.join(available)}"
    raise ValueError(msg)
