def __init__(self, config):
    super().__init__(config)
    self.roberta = XLMRobertaModel(config)
    self.transformation = nn.Linear(config.hidden_size, config.project_dim)
    self.post_init()
