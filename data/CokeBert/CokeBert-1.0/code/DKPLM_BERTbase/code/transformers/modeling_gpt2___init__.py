def __init__(self, config):
    super(GPT2DoubleHeadsModel, self).__init__(config)
    self.transformer = GPT2Model(config)
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.multiple_choice_head = SequenceSummary(config)
    self.init_weights()
