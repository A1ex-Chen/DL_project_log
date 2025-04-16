def __init__(self, embed_dim: int, query_heads: int=8, kv_heads: int=4,
    dropout: float=0.1, bias: bool=True, layer_norm: bool=True,
    layer_norm_eps: float=1e-05, gamma_init: float=1.0, linear_groups: int=
    1, *args, **kwargs):
    super().__init__()
    self.query_heads = query_heads
    self.kv_heads = kv_heads
    self.dropout = dropout
    self.layer_norm = layer_norm
    self.gamma_init = gamma_init
    if self.query_heads % self.kv_heads != 0:
        raise ValueError(
            f'query_heads ({query_heads}) must be divisible by kv_heads ({kv_heads})'
            )
    elif embed_dim % self.query_heads != 0 or embed_dim % self.kv_heads != 0:
        raise ValueError(
            f'embed_dim ({embed_dim}) must be divisible by query_heads ({query_heads}) and kv_heads ({kv_heads})'
            )
    head_dim = embed_dim // query_heads
    if not head_dim % 8 == 0:
        raise ValueError(
            f'head_dim (embed_dim / num_heads = {head_dim}) must be divisible by 8'
            )
    if not head_dim <= 128:
        raise ValueError(
            f'head_dim (embed_dim / num_heads = {head_dim}) must be <= 128')
    self.q_proj = BitLinear(embed_dim, embed_dim, *args, bias=bias, **kwargs)
    kv_embed_dim = embed_dim // query_heads * kv_heads
    self.k_proj = BitLinear(embed_dim, kv_embed_dim, *args, bias=bias, **kwargs
        )
    self.v_proj = BitLinear(embed_dim, kv_embed_dim, *args, bias=bias, **kwargs
        )
    self.norm: Optional[nn.LayerNorm] = None
    if layer_norm:
        self.norm = nn.LayerNorm(kv_embed_dim, eps=layer_norm_eps)
    self.out_proj = BitLinear(kv_embed_dim, embed_dim, bias=bias)
    self._reset_parameters()
