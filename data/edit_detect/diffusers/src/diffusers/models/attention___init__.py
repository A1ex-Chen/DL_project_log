def __init__(self, dim: int, dim_out: Optional[int]=None, mult: int=4,
    dropout: float=0.0, activation_fn: str='geglu', final_dropout: bool=
    False, inner_dim=None, bias: bool=True):
    super().__init__()
    if inner_dim is None:
        inner_dim = int(dim * mult)
    dim_out = dim_out if dim_out is not None else dim
    if activation_fn == 'gelu':
        act_fn = GELU(dim, inner_dim, bias=bias)
    if activation_fn == 'gelu-approximate':
        act_fn = GELU(dim, inner_dim, approximate='tanh', bias=bias)
    elif activation_fn == 'geglu':
        act_fn = GEGLU(dim, inner_dim, bias=bias)
    elif activation_fn == 'geglu-approximate':
        act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
    self.net = nn.ModuleList([])
    self.net.append(act_fn)
    self.net.append(nn.Dropout(dropout))
    self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
    if final_dropout:
        self.net.append(nn.Dropout(dropout))
