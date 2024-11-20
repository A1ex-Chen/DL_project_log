def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
    super().__init__()
    self.aap = nn.AdaptiveAvgPool2d(1)
    self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)
    self.flat = nn.Flatten()
