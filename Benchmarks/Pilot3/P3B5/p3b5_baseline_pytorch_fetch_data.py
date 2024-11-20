def fetch_data(gParameters):
    """Download and unpack data

    Args:
        gParameters: parameters from candle

    Returns:
        path to where the data is located
    """
    path = gParameters['data_url']
    fpath = candle.fetch_file(path + gParameters['train_data'], 'Pilot3',
        unpack=True)
    return fpath
