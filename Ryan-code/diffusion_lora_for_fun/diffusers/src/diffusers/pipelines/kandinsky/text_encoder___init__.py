def __init__(self, config, *args, **kwargs):
    super().__init__(config, *args, **kwargs)
    self.transformer = XLMRobertaModel(config)
    self.LinearTransformation = torch.nn.Linear(in_features=config.
        transformerDimensions, out_features=config.numDims)
