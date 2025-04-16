def __init__(self, config, transformer, encoder):
    super().__init__()
    self.num_labels = config.num_labels
    self.mmbt = MMBTModel(config, transformer, encoder)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size, config.num_labels)
