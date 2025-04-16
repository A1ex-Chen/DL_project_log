def __init__(self, c1, c2, g=1, e=4, dw_k=3, act='relu', downsample=False,
    ignore_group=False):
    super(ShuffleUnit, self).__init__()
    self.downsample = downsample
    mid_channels = c2 // e
    if downsample:
        c2 -= c1
    if c1 != c2:
        raise ValueError(
            'Input and output channel count of ShuffleUnit does not match')
    self.compress_conv_bn_act1 = ConvBnAct(c1=c1, c2=mid_channels, k=1, g=1 if
        ignore_group else g, act=act)
    self.c_shuffle = ChannelShuffle(channels=mid_channels, groups=g)
    self.dw_conv_bn2 = DWConv(c1=mid_channels, c2=mid_channels, k=dw_k, s=2 if
        self.downsample else 1, act=False)
    self.expand_conv_bn3 = ConvBnAct(c1=mid_channels, c2=c2, k=1, g=g, act=
        False)
    if downsample:
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    self.activ = get_activation(act)
