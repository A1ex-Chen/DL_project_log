def __init__(self, config: LlamaConfig):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.self_attn = LlamaAttention(config=config)
    mlp_kwargs = {'config': config, 'hidden_size': config.hidden_size,
        'intermediate_size': config.intermediate_size, 'hidden_act': config
        .hidden_act}
    valid_params = set(inspect.signature(LlamaMLP.__init__).parameters.keys()
        ) - {'self'}
    mlp_kwargs = {k: v for k, v in mlp_kwargs.items() if k in valid_params}
    self.mlp = LlamaMLP(**mlp_kwargs)
    self.input_layernorm = MultiwayNetwork(module_provider=partial(
        LlamaRMSNorm, hidden_size=config.hidden_size, eps=config.rms_norm_eps))
    self.post_attention_layernorm = MultiwayNetwork(module_provider=partial
        (LlamaRMSNorm, hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        )
