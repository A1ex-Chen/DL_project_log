def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=100):
    super(ResNeXt, self).__init__()
    self.cardinality = cardinality
    self.bottleneck_width = bottleneck_width
    self.in_planes = 64
    self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.layer1 = self._make_layer(num_blocks[0], 1)
    self.layer2 = self._make_layer(num_blocks[1], 2)
    self.layer3 = self._make_layer(num_blocks[2], 2)
    self.linear = nn.Linear(cardinality * bottleneck_width * 8, num_classes)
