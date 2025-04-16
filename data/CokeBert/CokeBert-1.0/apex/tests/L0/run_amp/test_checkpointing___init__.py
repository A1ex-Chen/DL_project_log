def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 3, 1, 1)
    self.bn1 = nn.BatchNorm2d(6)
    self.param = nn.Parameter(torch.randn(1))
