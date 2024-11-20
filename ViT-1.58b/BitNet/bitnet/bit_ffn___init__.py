def __init__(self, dim: int, dim_out: Optional[int]=None, mult: int=4, glu:
    bool=False, glu_mult_bias: bool=False, swish: bool=False, post_act_ln:
    bool=False, dropout: float=0.0, no_bias: bool=False, zero_init_output:
    bool=False, *args, **kwargs):
    super().__init__()
    inner_dim = int(dim * mult)
    dim_out = default(dim_out, dim)
    if swish:
        activation = nn.SiLU()
    else:
        activation = nn.GELU()
    if glu:
        project_in = GLU(dim, inner_dim, activation, mult_bias=glu_mult_bias)
    else:
        project_in = nn.Sequential(BitLinear(dim, inner_dim, *args, bias=
            not no_bias, **kwargs), activation)
    self.ff = nn.Sequential(project_in, nn.LayerNorm(inner_dim) if
        post_act_ln else None, nn.Dropout(dropout), BitLinear(inner_dim,
        dim_out, *args, bias=not no_bias, **kwargs))
    if zero_init_output:
        init_zero_(self.ff[-1])
