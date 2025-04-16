def __init__(self, config: LDMBertConfig):
    super().__init__(config)
    self.model = LDMBertEncoder(config)
    self.to_logits = nn.Linear(config.hidden_size, config.vocab_size)
