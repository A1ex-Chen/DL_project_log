def __init__(self, num_channels, time_embed_dim, context_dim=None,
    norm_groups=32, head_dim=64, expansion_ratio=4):
    super().__init__()
    self.in_norm = Kandinsky3ConditionalGroupNorm(norm_groups, num_channels,
        time_embed_dim)
    self.attention = Attention(num_channels, context_dim or num_channels,
        dim_head=head_dim, out_dim=num_channels, out_bias=False)
    hidden_channels = expansion_ratio * num_channels
    self.out_norm = Kandinsky3ConditionalGroupNorm(norm_groups,
        num_channels, time_embed_dim)
    self.feed_forward = nn.Sequential(nn.Conv2d(num_channels,
        hidden_channels, kernel_size=1, bias=False), nn.SiLU(), nn.Conv2d(
        hidden_channels, num_channels, kernel_size=1, bias=False))
