def __init__(self, num_attention_heads: int=16, attention_head_dim: int=88,
    in_channels: int=320, out_channels: Optional[int]=None, num_layers: int
    =1, cross_attention_dim: Optional[int]=None):
    super().__init__()
    self.num_attention_heads = num_attention_heads
    self.attention_head_dim = attention_head_dim
    inner_dim = num_attention_heads * attention_head_dim
    self.inner_dim = inner_dim
    self.in_channels = in_channels
    self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels,
        eps=1e-06)
    self.proj_in = nn.Linear(in_channels, inner_dim)
    self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(
        inner_dim, num_attention_heads, attention_head_dim,
        cross_attention_dim=cross_attention_dim) for d in range(num_layers)])
    time_mix_inner_dim = inner_dim
    self.temporal_transformer_blocks = nn.ModuleList([
        TemporalBasicTransformerBlock(inner_dim, time_mix_inner_dim,
        num_attention_heads, attention_head_dim, cross_attention_dim=
        cross_attention_dim) for _ in range(num_layers)])
    time_embed_dim = in_channels * 4
    self.time_pos_embed = TimestepEmbedding(in_channels, time_embed_dim,
        out_dim=in_channels)
    self.time_proj = Timesteps(in_channels, True, 0)
    self.time_mixer = AlphaBlender(alpha=0.5, merge_strategy=
        'learned_with_images')
    self.out_channels = in_channels if out_channels is None else out_channels
    self.proj_out = nn.Linear(inner_dim, in_channels)
    self.gradient_checkpointing = False
