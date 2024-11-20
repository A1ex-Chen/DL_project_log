def __init__(self, config):
    super().__init__(config)
    self.num_labels = config.num_labels
    self.transformer = GPT2Model(config)
    self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)
    self.init_weights()
