def __init__(self, dim: int, num_attention_heads: int, attention_head_dim:
    int, dropout: float=0.0, cross_attention_dim: Optional[int]=None,
    attention_bias: bool=False, upcast_attention: bool=False, temb_channels:
    int=768, add_self_attention: bool=False, cross_attention_norm: Optional
    [str]=None, group_size: int=32):
    super().__init__()
    self.add_self_attention = add_self_attention
    if add_self_attention:
        self.norm1 = AdaGroupNorm(temb_channels, dim, max(1, dim // group_size)
            )
        self.attn1 = Attention(query_dim=dim, heads=num_attention_heads,
            dim_head=attention_head_dim, dropout=dropout, bias=
            attention_bias, cross_attention_dim=None, cross_attention_norm=None
            )
    self.norm2 = AdaGroupNorm(temb_channels, dim, max(1, dim // group_size))
    self.attn2 = Attention(query_dim=dim, cross_attention_dim=
        cross_attention_dim, heads=num_attention_heads, dim_head=
        attention_head_dim, dropout=dropout, bias=attention_bias,
        upcast_attention=upcast_attention, cross_attention_norm=
        cross_attention_norm)
