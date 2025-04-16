def __init__(self, config, *inputs, **kwargs):
    super().__init__(config, *inputs, **kwargs)
    self.transformer = TFCTRLMainLayer(config, name='transformer')
    self.lm_head = TFCTRLLMHead(config, self.transformer.w, name='lm_head')
