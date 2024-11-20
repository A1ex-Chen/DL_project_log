def __init__(self, config, *inputs, **kwargs):
    super().__init__(config, *inputs, **kwargs)
    self.transformer = TFFlaubertMainLayer(config, name='transformer')
