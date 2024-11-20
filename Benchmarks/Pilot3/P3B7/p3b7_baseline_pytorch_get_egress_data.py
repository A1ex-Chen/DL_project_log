def get_egress_data(tasks):
    """Initialize egress tokenized data loaders

    Args:
        args: CANDLE ArgumentStruct
        tasks: dictionary of the number of classes for each task

    Returns:
        train and valid data
    """
    train_data = Egress('./data', 'train')
    valid_data = Egress('./data', 'valid')
    return train_data, valid_data
