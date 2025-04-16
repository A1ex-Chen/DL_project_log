def __init__(self, config: ShikraConfig):
    super(LlamaForCausalLM, self).__init__(config)
    self.model = ShikraLlamaModel(config)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.post_init()
