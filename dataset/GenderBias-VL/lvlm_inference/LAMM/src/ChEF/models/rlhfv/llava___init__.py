def __init__(self, config, tune_clip=False):
    super(LlamaForCausalLM, self).__init__(config)
    self.model = LlavaLlamaModel(config, tune_clip=tune_clip)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.post_init()
