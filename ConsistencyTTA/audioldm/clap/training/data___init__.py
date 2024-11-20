def __init__(self, input_filename, transforms, img_key, caption_key, sep='\t'):
    logging.debug(f'Loading csv data from {input_filename}.')
    df = pd.read_csv(input_filename, sep=sep)
    self.images = df[img_key].tolist()
    self.captions = df[caption_key].tolist()
    self.transforms = transforms
    logging.debug('Done loading data.')
