def __init__(self, config: MPTConfig):
    super().__init__(config)
    if not config.tie_word_embeddings:
        raise ValueError('MPTForCausalLM only supports tied word embeddings')
    print(f'Instantiating an MPTForCausalLM model from {__file__}')
    self.transformer = MPTModel(config)
    for child in self.transformer.children():
        if isinstance(child, torch.nn.ModuleList):
            continue
        if isinstance(child, torch.nn.Module):
            child._fsdp_wrap = True
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
