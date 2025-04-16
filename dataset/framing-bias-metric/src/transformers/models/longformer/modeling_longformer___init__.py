def __init__(self, config):
    super().__init__(config)
    self.longformer = LongformerModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size, 1)
    self.init_weights()
