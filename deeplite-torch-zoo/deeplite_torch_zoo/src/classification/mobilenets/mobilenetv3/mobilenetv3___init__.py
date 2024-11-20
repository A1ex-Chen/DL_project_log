def __init__(self, cfgs, mode, num_classes=1000, width_mult=1.0):
    super(MobileNetV3, self).__init__()
    self.cfgs = cfgs
    assert mode in ['large', 'small']
    input_channel = round_channels(16 * width_mult, 8)
    layers = [conv_3x3_bn(3, input_channel, 2)]
    block = InvertedResidual
    for k, t, c, use_se, use_hs, s in self.cfgs:
        output_channel = round_channels(c * width_mult, 8)
        exp_size = round_channels(input_channel * t, 8)
        layers.append(block(input_channel, exp_size, output_channel, k, s,
            use_se, use_hs))
        input_channel = output_channel
    self.features = nn.Sequential(*layers)
    self.conv = conv_1x1_bn(input_channel, exp_size)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    output_channel = {'large': 1280, 'small': 1024}
    output_channel = round_channels(output_channel[mode] * width_mult, 8
        ) if width_mult > 1.0 else output_channel[mode]
    self.classifier = nn.Sequential(nn.Linear(exp_size, output_channel),
        h_swish(), nn.Dropout(0.2), nn.Linear(output_channel, num_classes))
    self._initialize_weights()
