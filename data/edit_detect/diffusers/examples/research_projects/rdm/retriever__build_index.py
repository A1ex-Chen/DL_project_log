@staticmethod
def _build_index(config: IndexConfig, dataset: Dataset=None, model=None,
    feature_extractor: CLIPFeatureExtractor=None):
    dataset = dataset or load_dataset(config.dataset_name)
    dataset = dataset[config.dataset_set]
    index = Index(config, dataset)
    index.build_index(model=model, feature_extractor=feature_extractor)
    return index
