def __init__(self, config):
    super(TFTransfoXLLMHeadModel, self).__init__(config)
    self.transformer = TFTransfoXLMainLayer(config, name='transformer')
    self.sample_softmax = config.sample_softmax
    if config.sample_softmax > 0:
        raise NotImplementedError
    else:
        self.crit = TFAdaptiveSoftmaxMask(config.n_token, config.d_embed,
            config.d_model, config.cutoffs, div_val=config.div_val, name='crit'
            )
