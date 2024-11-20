def __init__(self, num_attention_heads: int=16, attention_head_dim: int=88,
    in_channels: Optional[int]=None, num_layers: int=1, dropout: float=0.0,
    norm_num_groups: int=32, cross_attention_dim: Optional[int]=None,
    attention_bias: bool=False, sample_size: Optional[int]=None,
    num_vector_embeds: Optional[int]=None, activation_fn: str='geglu',
    num_embeds_ada_norm: Optional[int]=None):
    super().__init__()
    self.transformers = nn.ModuleList([Transformer2DModel(
        num_attention_heads=num_attention_heads, attention_head_dim=
        attention_head_dim, in_channels=in_channels, num_layers=num_layers,
        dropout=dropout, norm_num_groups=norm_num_groups,
        cross_attention_dim=cross_attention_dim, attention_bias=
        attention_bias, sample_size=sample_size, num_vector_embeds=
        num_vector_embeds, activation_fn=activation_fn, num_embeds_ada_norm
        =num_embeds_ada_norm) for _ in range(2)])
    self.mix_ratio = 0.5
    self.condition_lengths = [77, 257]
    self.transformer_index_for_condition = [1, 0]
