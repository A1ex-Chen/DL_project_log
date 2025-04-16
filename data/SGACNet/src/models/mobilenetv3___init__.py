def __init__(self, block_args, num_classes=1000, in_chans=3, stem_size=16,
    num_features=1280, head_bias=True, channel_multiplier=1.0, pad_type='',
    act_layer=HardSwish, drop_rate=0.0, drop_connect_rate=0.0, se_kwargs=
    None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, weight_init='goog'):
    super(MobileNetV3, self).__init__()
    self.drop_rate = drop_rate
    stem_size = round_channels(stem_size, channel_multiplier)
    self.conv_stem = select_conv2d(in_chans, stem_size, 3, stride=2,
        padding=pad_type)
    self.bn1 = nn.BatchNorm2d(stem_size, **norm_kwargs)
    self.act1 = act_layer(inplace=True)
    in_chs = stem_size
    builder = EfficientNetBuilder(channel_multiplier, pad_type=pad_type,
        act_layer=act_layer, se_kwargs=se_kwargs, norm_layer=norm_layer,
        norm_kwargs=norm_kwargs, drop_connect_rate=drop_connect_rate)
    self.blocks = nn.Sequential(*builder(in_chs, block_args))
    in_chs = builder.in_chs
    self.global_pool = nn.AdaptiveAvgPool2d(1)
    self.conv_head = select_conv2d(in_chs, num_features, 1, padding=
        pad_type, bias=head_bias)
    self.act2 = act_layer(inplace=True)
    self.classifier = nn.Linear(num_features, num_classes)
    for m in self.modules():
        if weight_init == 'goog':
            initialize_weight_goog(m)
        else:
            initialize_weight_default(m)
