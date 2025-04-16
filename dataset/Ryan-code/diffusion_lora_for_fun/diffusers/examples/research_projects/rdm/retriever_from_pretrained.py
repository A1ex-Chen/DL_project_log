@classmethod
def from_pretrained(cls, retriever_name_or_path: str, index: Index=None,
    dataset: Dataset=None, model=None, feature_extractor:
    CLIPFeatureExtractor=None, **kwargs):
    config = kwargs.pop('config', None) or IndexConfig.from_pretrained(
        retriever_name_or_path, **kwargs)
    return cls(config, index=index, dataset=dataset, model=model,
        feature_extractor=feature_extractor)
