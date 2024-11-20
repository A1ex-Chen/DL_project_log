def __init__(self, config):
    super(LlamaForCausalLM, self).__init__(config)
    self.model = LlavaLlamaModel(config)
    self.pretraining_tp = config.pretraining_tp
    self.vocab_size = config.vocab_size
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.post_init()
