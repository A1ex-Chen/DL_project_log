def __init__(self, config):
    super(MPTForCausalLM, self).__init__(config)
    if not config.tie_word_embeddings:
        raise ValueError('MPTForCausalLM only supports tied word embeddings')
    self.transformer = LlavaMPTModel(config)
    self.logit_scale = None
    if config.logit_scale is not None:
        logit_scale = config.logit_scale
        if isinstance(logit_scale, str):
            if logit_scale == 'inv_sqrt_d_model':
                logit_scale = 1 / math.sqrt(config.d_model)
            else:
                raise ValueError(
                    f"logit_scale={logit_scale!r} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'."
                    )
        self.logit_scale = logit_scale
