def fetch_data(gParameters):
    """Downloads and decompresses the data if not locally available.
    Since the training data depends on the model definition it is not loaded,
    instead the local path where the raw data resides is returned
    """
    path = gParameters['data_url']
    fpath = candle.fetch_file(path + gParameters['train_data'], 'Pilot3',
        unpack=True)
    return fpath
