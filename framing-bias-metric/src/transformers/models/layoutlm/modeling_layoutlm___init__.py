def __init__(self, config):
    super().__init__(config)
    self.num_labels = config.num_labels
    self.layoutlm = LayoutLMModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    self.init_weights()
