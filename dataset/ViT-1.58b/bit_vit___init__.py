def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0,
    proj_drop=0.0):
    super().__init__()
    assert dim % num_heads == 0, 'dim should be divisible by num_heads'
    self.num_heads = num_heads
    head_dim = dim // num_heads
    self.scale = head_dim ** -0.5
    self.qkv = BitLinear(dim, dim * 3, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = BitLinear(dim, dim)
    self.proj_drop = nn.Dropout(proj_drop)
