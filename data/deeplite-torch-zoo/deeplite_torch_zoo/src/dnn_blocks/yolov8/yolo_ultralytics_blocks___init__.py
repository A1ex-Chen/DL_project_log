def __init__(self, c1, c2, num_heads, num_layers, act='hardswish'):
    super().__init__()
    Conv_ = functools.partial(ConvBnAct, act=act)
    self.conv = None
    if c1 != c2:
        self.conv = Conv_(c1, c2)
    self.linear = nn.Linear(c2, c2)
    self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in
        range(num_layers)))
    self.c2 = c2
