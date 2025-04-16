def __init__(self, config, language_hidden_size):
    super().__init__(config)
    self.config = config
    self.encoder = MplugOwlVisualAbstractorEncoder(config)
    self.visual_fc = torch.nn.Linear(config.hidden_size, language_hidden_size)
    self.query_embeds = torch.nn.Parameter(torch.randn(1, config.
        num_learnable_queries, config.hidden_size))
    self.vit_eos = torch.nn.Parameter(torch.randn(1, 1, language_hidden_size))
    self.post_init()
