def __init__(self, config):
    super().__init__(config)
    self.model = OPTModel(config)
    self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size,
        bias=False)
    self.post_init()
