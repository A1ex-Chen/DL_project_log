def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2, act='relu'):
    super(GhostNetV2, self).__init__()
    self.cfgs = cfgs
    self.dropout = dropout
    output_channel = round_channels(16 * width, 4)
    self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(output_channel)
    self.act1 = get_activation(act)
    input_channel = output_channel
    stages = []
    layer_id = 0
    for cfg in self.cfgs:
        layers = []
        for k, exp_size, c, se_ratio, s in cfg:
            output_channel = round_channels(c * width, 4)
            hidden_channel = round_channels(exp_size * width, 4)
            layers.append(GhostBottleneckV2(input_channel, output_channel,
                hidden_channel, k, s, se_ratio=se_ratio, layer_id=layer_id,
                act=act))
            input_channel = output_channel
            layer_id += 1
        stages.append(nn.Sequential(*layers))
    output_channel = round_channels(exp_size * width, 4)
    stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
    input_channel = output_channel
    self.blocks = nn.Sequential(*stages)
    output_channel = 1280
    self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
    self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias
        =True)
    self.act2 = get_activation(act)
    self.classifier = nn.Linear(output_channel, num_classes)
