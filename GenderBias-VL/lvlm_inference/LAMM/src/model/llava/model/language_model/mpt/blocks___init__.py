def __init__(self, d_model: int, n_heads: int, expansion_ratio: int,
    attn_config: Dict={'attn_type': 'multihead_attention', 'attn_pdrop': 
    0.0, 'attn_impl': 'triton', 'qk_ln': False, 'clip_qkv': None,
    'softmax_scale': None, 'prefix_lm': False, 'attn_uses_sequence_id': 
    False, 'alibi': False, 'alibi_bias_max': 8}, resid_pdrop: float=0.0,
    norm_type: str='low_precision_layernorm', verbose: int=0, device:
    Optional[str]=None, **kwargs):
    del kwargs
    super().__init__()
    norm_class = NORM_CLASS_REGISTRY[norm_type.lower()]
    attn_class = ATTN_CLASS_REGISTRY[attn_config['attn_type']]
    self.norm_1 = norm_class(d_model, device=device)
    self.attn = attn_class(attn_impl=attn_config['attn_impl'], clip_qkv=
        attn_config['clip_qkv'], qk_ln=attn_config['qk_ln'], softmax_scale=
        attn_config['softmax_scale'], attn_pdrop=attn_config['attn_pdrop'],
        d_model=d_model, n_heads=n_heads, verbose=verbose, device=device)
    self.norm_2 = norm_class(d_model, device=device)
    self.ffn = MPTMLP(d_model=d_model, expansion_ratio=expansion_ratio,
        device=device)
    self.resid_attn_dropout = nn.Dropout(resid_pdrop)
    self.resid_ffn_dropout = nn.Dropout(resid_pdrop)
