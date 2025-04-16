def __init__(self, num_classes=100):
    super(MobileNetV2, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=
        False)
    self.bn1 = nn.BatchNorm2d(32)
    self.layers = self._make_layers(in_planes=32)
    self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0,
        bias=False)
    self.bn2 = nn.BatchNorm2d(1280)
    self.linear = nn.Linear(1280, num_classes)
