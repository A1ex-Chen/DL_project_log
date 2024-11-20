def __init__(self, in_features: int, out_features: int):
    super().__init__()
    self.scale_bias = nn.Linear(in_features, out_features * 2, bias=False)
