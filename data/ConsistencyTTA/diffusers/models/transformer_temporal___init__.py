@register_to_config
def __init__(self, num_attention_heads: int=16, attention_head_dim: int=88,
    in_channels: Optional[int]=None, out_channels: Optional[int]=None,
    num_layers: int=1, dropout: float=0.0, norm_num_groups: int=32,
    cross_attention_dim: Optional[int]=None, attention_bias: bool=False,
    sample_size: Optional[int]=None, activation_fn: str='geglu',
    norm_elementwise_affine: bool=True, double_self_attention: bool=True):
    super().__init__()
    self.num_attention_heads = num_attention_heads
    self.attention_head_dim = attention_head_dim
    inner_dim = num_attention_heads * attention_head_dim
    self.in_channels = in_channels
    self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels
        =in_channels, eps=1e-06, affine=True)
    self.proj_in = nn.Linear(in_channels, inner_dim)
    self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(
        inner_dim, num_attention_heads, attention_head_dim, dropout=dropout,
        cross_attention_dim=cross_attention_dim, activation_fn=
        activation_fn, attention_bias=attention_bias, double_self_attention
        =double_self_attention, norm_elementwise_affine=
        norm_elementwise_affine) for d in range(num_layers)])
    self.proj_out = nn.Linear(inner_dim, in_channels)
