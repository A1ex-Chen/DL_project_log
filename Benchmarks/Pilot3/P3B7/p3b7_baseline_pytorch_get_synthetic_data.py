def get_synthetic_data(args):
    """Initialize data loaders

    Args:
        datapath: path to the synthetic data

    Returns:
        train and valid data
    """
    datapath = fetch_data(args)
    train_data = P3B3(datapath, 'train')
    valid_data = P3B3(datapath, 'test')
    return train_data, valid_data
