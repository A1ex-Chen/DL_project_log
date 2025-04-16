def __init__(self, dim_in: int, dim_out: int, bias: bool=True):
    super().__init__()
    self.proj = nn.Linear(dim_in, dim_out, bias=bias)
