@register_to_config
def __init__(self, prefix_length: int, prefix_inner_dim: int,
    prefix_hidden_dim: Optional[int]=None, vocab_size: int=50257,
    n_positions: int=1024, n_embd: int=768, n_layer: int=12, n_head: int=12,
    n_inner: Optional[int]=None, activation_function: str='gelu_new',
    resid_pdrop: float=0.1, embd_pdrop: float=0.1, attn_pdrop: float=0.1,
    layer_norm_epsilon: float=1e-05, initializer_range: float=0.02,
    scale_attn_weights: bool=True, use_cache: bool=True,
    scale_attn_by_inverse_layer_idx: bool=False, reorder_and_upcast_attn:
    bool=False):
    super().__init__()
    self.prefix_length = prefix_length
    if prefix_inner_dim != n_embd and prefix_hidden_dim is None:
        raise ValueError(
            f'`prefix_hidden_dim` cannot be `None` when `prefix_inner_dim`: {prefix_hidden_dim} and `n_embd`: {n_embd} are not equal.'
            )
    self.prefix_inner_dim = prefix_inner_dim
    self.prefix_hidden_dim = prefix_hidden_dim
    self.encode_prefix = nn.Linear(self.prefix_inner_dim, self.
        prefix_hidden_dim
        ) if self.prefix_hidden_dim is not None else nn.Identity()
    self.decode_prefix = nn.Linear(self.prefix_hidden_dim, n_embd
        ) if self.prefix_hidden_dim is not None else nn.Identity()
    gpt_config = GPT2Config(vocab_size=vocab_size, n_positions=n_positions,
        n_embd=n_embd, n_layer=n_layer, n_head=n_head, n_inner=n_inner,
        activation_function=activation_function, resid_pdrop=resid_pdrop,
        embd_pdrop=embd_pdrop, attn_pdrop=attn_pdrop, layer_norm_epsilon=
        layer_norm_epsilon, initializer_range=initializer_range,
        scale_attn_weights=scale_attn_weights, use_cache=use_cache,
        scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
        reorder_and_upcast_attn=reorder_and_upcast_attn)
    self.transformer = GPT2LMHeadModel(gpt_config)
