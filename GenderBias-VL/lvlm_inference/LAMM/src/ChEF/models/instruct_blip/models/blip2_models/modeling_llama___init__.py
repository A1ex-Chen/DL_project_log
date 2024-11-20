def __init__(self, config):
    super().__init__(config)
    self.num_labels = config.num_labels
    self.model = LlamaModel(config)
    self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
    self.post_init()
