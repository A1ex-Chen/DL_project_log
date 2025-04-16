def __init__(self):
    super(DummyNetWrapper, self).__init__()
    self.bn = nn.BatchNorm2d(3, affine=True)
    self.dn = DummyNet()
