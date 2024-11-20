def __init__(self, num_heads, embed_dim, dtype=None):
    super().__init__()
    self.dtype = dtype
    self.positional_embedding = nn.Parameter(torch.randn(1, embed_dim) / 
        embed_dim ** 0.5)
    self.k_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
    self.q_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
    self.v_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
    self.num_heads = num_heads
    self.dim_per_head = embed_dim // self.num_heads
