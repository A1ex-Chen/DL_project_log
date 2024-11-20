def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0,
    context_dim=None, no_context=False):
    super().__init__()
    if no_context:
        context_dim = None
    self.in_channels = in_channels
    inner_dim = n_heads * d_head
    self.norm = Normalize(in_channels)
    self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=
        1, padding=0)
    self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(
        inner_dim, n_heads, d_head, dropout=dropout, context_dim=
        context_dim) for d in range(depth)])
    self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels,
        kernel_size=1, stride=1, padding=0))
