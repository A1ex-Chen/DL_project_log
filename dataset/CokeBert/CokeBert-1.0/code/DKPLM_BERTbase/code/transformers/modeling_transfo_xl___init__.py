def __init__(self, config):
    super(TransfoXLLMHeadModel, self).__init__(config)
    self.transformer = TransfoXLModel(config)
    self.sample_softmax = config.sample_softmax
    if config.sample_softmax > 0:
        self.out_layer = nn.Linear(config.d_model, config.n_token)
        self.sampler = LogUniformSampler(config.n_token, config.sample_softmax)
    else:
        self.crit = ProjectedAdaptiveLogSoftmax(config.n_token, config.
            d_embed, config.d_model, config.cutoffs, div_val=config.div_val)
    self.init_weights()
