def __init__(self, dim: int, eps: float=1e-06):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))
