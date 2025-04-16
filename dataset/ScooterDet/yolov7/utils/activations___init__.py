def __init__(self, c1, k=3):
    super().__init__()
    self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
    self.bn = nn.BatchNorm2d(c1)
