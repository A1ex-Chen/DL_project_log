def __init__(self, block, nblocks, growth_rate=12, reduction=0.5,
    num_classes=100):
    super(DenseNet, self).__init__()
    self.growth_rate = growth_rate
    num_planes = 2 * growth_rate
    self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)
    self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
    num_planes += nblocks[0] * growth_rate
    out_planes = int(math.floor(num_planes * reduction))
    self.trans1 = Transition(num_planes, out_planes)
    num_planes = out_planes
    self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
    num_planes += nblocks[1] * growth_rate
    out_planes = int(math.floor(num_planes * reduction))
    self.trans2 = Transition(num_planes, out_planes)
    num_planes = out_planes
    self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
    num_planes += nblocks[2] * growth_rate
    out_planes = int(math.floor(num_planes * reduction))
    self.trans3 = Transition(num_planes, out_planes)
    num_planes = out_planes
    self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
    num_planes += nblocks[3] * growth_rate
    self.bn = nn.BatchNorm2d(num_planes)
    self.linear = nn.Linear(num_planes, num_classes)
