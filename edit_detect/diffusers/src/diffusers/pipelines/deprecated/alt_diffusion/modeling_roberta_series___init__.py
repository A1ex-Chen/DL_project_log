def __init__(self, config):
    super().__init__(config)
    self.roberta = XLMRobertaModel(config)
    self.transformation = nn.Linear(config.hidden_size, config.project_dim)
    self.has_pre_transformation = getattr(config, 'has_pre_transformation',
        False)
    if self.has_pre_transformation:
        self.transformation_pre = nn.Linear(config.hidden_size, config.
            project_dim)
        self.pre_LN = nn.LayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
    self.post_init()
