def __init__(self, num_classes=100):
    super(MobileNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=
        False)
    self.bn1 = nn.BatchNorm2d(32)
    self.layers = self._make_layers(in_planes=32)
    self.linear = nn.Linear(1024, num_classes)
