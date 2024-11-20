def __init__(self, embedding_dim: int, out_dim: int, num_groups: int,
    act_fn: Optional[str]=None, eps: float=1e-05):
    super().__init__()
    self.num_groups = num_groups
    self.eps = eps
    self.act = None
    if act_fn == 'swish':
        self.act = lambda x: F.silu(x)
    elif act_fn == 'mish':
        self.act = nn.Mish()
    elif act_fn == 'silu':
        self.act = nn.SiLU()
    elif act_fn == 'gelu':
        self.act = nn.GELU()
    self.linear = nn.Linear(embedding_dim, out_dim * 2)
