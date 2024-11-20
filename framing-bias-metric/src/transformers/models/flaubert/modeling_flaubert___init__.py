def __init__(self, config):
    super().__init__(config)
    self.transformer = FlaubertModel(config)
    self.init_weights()
