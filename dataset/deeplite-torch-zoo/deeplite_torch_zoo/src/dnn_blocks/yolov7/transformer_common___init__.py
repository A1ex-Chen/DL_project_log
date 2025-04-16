def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=
    4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=
    nn.SiLU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
    super().__init__()
    self.dim = dim
    self.num_heads = num_heads
    self.window_size = window_size
    self.shift_size = shift_size
    self.mlp_ratio = mlp_ratio
    assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
    self.norm1 = norm_layer(dim)
    self.attn = WindowAttention_v2(dim, window_size=(self.window_size, self
        .window_size), num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=
        attn_drop, proj_drop=drop, pretrained_window_size=(
        pretrained_window_size, pretrained_window_size))
    self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    self.norm2 = norm_layer(dim)
    mlp_hidden_dim = int(dim * mlp_ratio)
    self.mlp = Mlp_v2(in_features=dim, hidden_features=mlp_hidden_dim,
        act_layer=act_layer, drop=drop)
