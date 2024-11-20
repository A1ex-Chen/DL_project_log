def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0):
    super().__init__()
    self.self_attn = self_attn
    self.norm = WuerstchenLayerNorm(c, elementwise_affine=False, eps=1e-06)
    self.attention = Attention(query_dim=c, heads=nhead, dim_head=c //
        nhead, dropout=dropout, bias=True)
    self.kv_mapper = nn.Sequential(nn.SiLU(), nn.Linear(c_cond, c))
