def __init__(self, layrs, dilation=None, width_per_group=64, num_classes=
    1000, width_mult=1.0, inverted_residual_setting=None, norm_layer=nn.
    BatchNorm2d, round_nearest=8):
    """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
    super(MobileNetV2, self).__init__()
    block = InvertedResidual
    input_channel = 32
    last_channel = 1280
    self.dilation = 1
    groups = 1
    self.groups = groups
    self.base_width = width_per_group
    self.inplanes = 64
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    self._norm_layer = norm_layer
    if inverted_residual_setting is None:
        inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 
            3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]
            ]
    if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]
        ) != 4:
        raise ValueError(
            'inverted_residual_setting should be non-empty or a 4-element list, got {}'
            .format(inverted_residual_setting))
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    input_channel = _make_divisible(input_channel * width_mult, round_nearest)
    self.last_channel = _make_divisible(last_channel * max(1.0, width_mult),
        round_nearest)
    features = [ConvBNReLU(3, input_channel, stride=2)]
    for t, c, n, s in inverted_residual_setting:
        output_channel = _make_divisible(c * width_mult, round_nearest)
        for i in range(n):
            stride = s if i == 0 else 1
            features.append(block(input_channel, output_channel, stride,
                expand_ratio=t))
            input_channel = output_channel
    features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1)
        )
    self.features = nn.Sequential(*features)
    self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.
        last_channel, num_classes))
    self.down_2_channels_out = 64
    self.down_4_channels_out = 64
    self.down_8_channels_out = 128
    self.down_16_channels_out = 256
    self.down_32_channels_out = 512
    self.layer1 = self._make_layer(block, 64, layers[0], dilate=dilation[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=
        dilation[1])
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=
        dilation[2])
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=
        dilation[3])
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)
