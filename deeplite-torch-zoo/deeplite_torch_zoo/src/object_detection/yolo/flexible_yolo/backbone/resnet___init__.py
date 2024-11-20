def __init__(self, block, layers, cbam=False, dcn=False, drop_prob=0,
    base_channels=64, width=0.25, first_block_downsampling=False):
    super(Resnet, self).__init__()
    self.inplanes = int(base_channels * width)
    self.dcn = dcn
    self.cbam = cbam
    self.drop = drop_prob != 0
    self.conv1 = nn.Conv2d(3, int(base_channels * width), kernel_size=7,
        stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(int(base_channels * width))
    self.relu = nn.ReLU(inplace=True)
    self.out_channels = []
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    if self.drop:
        self.dropblock = LinearScheduler(DropBlock2D(drop_prob=drop_prob,
            block_size=5), start_value=0.0, stop_value=drop_prob, nr_steps=
            5000.0)
    ch = int(base_channels * width)
    self.layer1 = self._make_layer(block, ch, layers[0], stride=1 if not
        first_block_downsampling else 2)
    self.layer2 = self._make_layer(block, ch * 2, layers[1], stride=2, cbam
        =self.cbam)
    self.layer3 = self._make_layer(block, ch * 4, layers[2], stride=2, cbam
        =self.cbam, dcn=dcn)
    self.layer4 = self._make_layer(block, ch * 8, layers[3], stride=2, cbam
        =self.cbam, dcn=dcn)
    if self.dcn is not None:
        for m in self.modules():
            if isinstance(m, Bottleneck) or isinstance(m, BasicBlock):
                if hasattr(m, 'conv2_offset'):
                    constant_init(m.conv2_offset, 0)
    self.out_shape = self.out_channels[0] * 2, self.out_channels[1
        ] * 2, self.out_channels[2] * 2
    LOGGER.info('Backbone output channel: C3 {}, C4 {}, C5 {}'.format(self.
        out_channels[0] * 2, self.out_channels[1] * 2, self.out_channels[2] *
        2))
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    self.freeze_bn()
