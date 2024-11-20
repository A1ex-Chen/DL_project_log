def fetch_data(gParameters):
    """Downloads and decompresses the data if not locally available."""
    path = gParameters['data_url']
    fpath = candle.fetch_file(path + gParameters['train_data'], 'Pilot3',
        unpack=True)
    return fpath
