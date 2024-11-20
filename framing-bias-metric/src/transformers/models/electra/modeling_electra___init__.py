def __init__(self, config):
    super().__init__(config)
    self.electra = ElectraModel(config)
    self.sequence_summary = SequenceSummary(config)
    self.classifier = nn.Linear(config.hidden_size, 1)
    self.init_weights()
