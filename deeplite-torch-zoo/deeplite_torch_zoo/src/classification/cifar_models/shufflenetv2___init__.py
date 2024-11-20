def __init__(self, net_size=1, num_classes=100):
    super(ShuffleNetV2, self).__init__()
    out_channels = configs[net_size]['out_channels']
    num_blocks = configs[net_size]['num_blocks']
    self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=
        False)
    self.bn1 = nn.BatchNorm2d(24)
    self.in_channels = 24
    self.layer1 = self._make_layer(out_channels[0], num_blocks[0])
    self.layer2 = self._make_layer(out_channels[1], num_blocks[1])
    self.layer3 = self._make_layer(out_channels[2], num_blocks[2])
    self.conv2 = nn.Conv2d(out_channels[2], out_channels[3], kernel_size=1,
        stride=1, padding=0, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels[3])
    self.linear = nn.Linear(out_channels[3], num_classes)
