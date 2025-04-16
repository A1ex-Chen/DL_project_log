def __init__(self, config, *inputs, **kwargs):
    super().__init__(config, *inputs, **kwargs)
    self.transformer = XLMModel(config)
    self.sequence_summary = SequenceSummary(config)
    self.logits_proj = nn.Linear(config.num_labels, 1)
    self.init_weights()
