def __init__(self, config):
    super(OpenAIGPTDoubleHeadsModel, self).__init__(config)
    self.transformer = OpenAIGPTModel(config)
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.multiple_choice_head = SequenceSummary(config)
    self.init_weights()
