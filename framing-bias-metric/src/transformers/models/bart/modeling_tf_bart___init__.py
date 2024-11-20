def __init__(self, config, *inputs, **kwargs):
    super().__init__(config, *inputs, **kwargs)
    self.model = TFBartModel(config, name='model')
    self.use_cache = config.use_cache
    self.final_logits_bias = self.add_weight(name='/final_logits_bias',
        shape=[1, config.vocab_size], initializer='zeros', trainable=False)
