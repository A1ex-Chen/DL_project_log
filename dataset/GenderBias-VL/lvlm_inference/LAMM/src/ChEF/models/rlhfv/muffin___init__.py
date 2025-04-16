def __init__(self, config, mm_vision_tower=None):
    super(LlamaForCausalLM, self).__init__(config)
    self.model = Beit3LlavaLlamaModel(config, mm_vision_tower=mm_vision_tower)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.post_init()
