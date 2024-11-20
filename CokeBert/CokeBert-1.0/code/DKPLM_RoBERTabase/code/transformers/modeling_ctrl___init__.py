def __init__(self, config):
    super(CTRLLMHeadModel, self).__init__(config)
    self.transformer = CTRLModel(config)
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)
    self.init_weights()
