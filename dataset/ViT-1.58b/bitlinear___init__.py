def __init__(self, in_features, out_features, bias=True):
    super(BitLinear, self).__init__(in_features, out_features, bias)
    self.norm = nn.LayerNorm(in_features)
    self.register_buffer('quant_weight', None)
    self.register_buffer('weight_scale', None)
