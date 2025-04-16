def __init__(self, layers, block, zero_init_residual=False, groups=1,
    width_per_group=64, replace_stride_with_dilation=None, dilation=None,
    norm_layer=None, input_channels=3, activation=nn.ReLU(inplace=True)):
    super(ResNet, self).__init__()
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    self._norm_layer = norm_layer
    self.inplanes = 64
    self.dilation = 1
    if replace_stride_with_dilation is None:
        replace_stride_with_dilation = [False, False, False]
    self.replace_stride_with_dilation = replace_stride_with_dilation
    if len(replace_stride_with_dilation) != 3:
        raise ValueError(
            'replace_stride_with_dilation should be None or a 3-element tuple, got {}'
            .format(replace_stride_with_dilation))
    if dilation is not None:
        dilation = dilation
        if len(dilation) != 4:
            raise ValueError(
                'dilation should be None or a 4-element tuple, got {}'.
                format(dilation))
    else:
        dilation = [1, 1, 1, 1]
    self.groups = groups
    self.base_width = width_per_group
    self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=7,
        stride=2, padding=3, bias=False)
    self.bn1 = norm_layer(self.inplanes)
    self.act = activation
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.down_2_channels_out = 64
    if self.replace_stride_with_dilation == [False, False, False]:
        self.down_4_channels_out = 64 * block.expansion
        self.down_8_channels_out = 128 * block.expansion
        self.down_16_channels_out = 256 * block.expansion
        self.down_32_channels_out = 512 * block.expansion
    elif self.replace_stride_with_dilation == [False, True, True]:
        self.down_4_channels_out = 64 * block.expansion
        self.down_8_channels_out = 512 * block.expansion
    self.layer1 = self._make_layer(block, 64, layers[0], dilate=dilation[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=
        dilation[1], replace_stride_with_dilation=
        replace_stride_with_dilation[0])
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=
        dilation[2], replace_stride_with_dilation=
        replace_stride_with_dilation[1])
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=
        dilation[3], replace_stride_with_dilation=
        replace_stride_with_dilation[2])
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=
                'relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    if zero_init_residual:
        for m in self.modules():
            nn.init.constant_(m.bn2.weight, 0)
