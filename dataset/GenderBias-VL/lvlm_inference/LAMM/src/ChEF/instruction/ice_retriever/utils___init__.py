def __init__(self, datalist: List, model_name=None, extractor=None) ->None:
    self.datalist = datalist
    if model_name is None and extractor is None:
        raise ValueError('model_name and extractor could not both be None')
    if extractor is not None:
        self.extractor = extractor
    else:
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
    self.encode_dataset = []
    self.init_dataset()
    self.datalist_length = len(self.encode_dataset)
