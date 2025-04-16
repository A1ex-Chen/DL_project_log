def __init__(self, in_features, out_features):
    super().__init__()
    self.scale_bias = nn.Linear(in_features, out_features * 2, bias=False)
