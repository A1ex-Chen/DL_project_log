def __init__(self, config: IndexConfig, index: Index=None, dataset: Dataset
    =None, model=None, feature_extractor: CLIPFeatureExtractor=None):
    self.config = config
    self.index = index or self._build_index(config, dataset, model=model,
        feature_extractor=feature_extractor)
