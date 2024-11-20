def __init__(self, config):
    super().__init__(config)
    self.transformer = TransfoXLModel(config)
    self.sample_softmax = config.sample_softmax
    assert self.sample_softmax <= 0, 'Sampling from the softmax is not implemented yet. Please look at issue: #3310: https://github.com/huggingface/transformers/issues/3310'
    self.crit = ProjectedAdaptiveLogSoftmax(config.vocab_size, config.
        d_embed, config.d_model, config.cutoffs, div_val=config.div_val)
    self.init_weights()
