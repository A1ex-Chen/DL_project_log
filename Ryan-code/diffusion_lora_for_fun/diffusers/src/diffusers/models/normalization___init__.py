def __init__(self, dim, eps: float=1e-05, elementwise_affine: bool=True,
    bias: bool=True):
    super().__init__()
    self.eps = eps
    if isinstance(dim, numbers.Integral):
        dim = dim,
    self.dim = torch.Size(dim)
    if elementwise_affine:
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
    else:
        self.weight = None
        self.bias = None
