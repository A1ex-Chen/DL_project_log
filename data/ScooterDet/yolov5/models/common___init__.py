def __init__(self, c1, c2, k=1, s=1, p=None, g=1, dropout_p=0.0):
    super().__init__()
    c_ = 1280
    self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.drop = nn.Dropout(p=dropout_p, inplace=True)
    self.linear = nn.Linear(c_, c2)
