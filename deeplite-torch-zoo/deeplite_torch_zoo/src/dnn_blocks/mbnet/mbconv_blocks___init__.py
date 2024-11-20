def __init__(self, c1, c2, e=1.0, k=3, stride=1, act='relu', se_ratio=None,
    channel_divisor=1, shortcut=True):
    super().__init__()
    assert stride in (1, 2)
    self.shortcut = shortcut
    if e == 1.0:
        self.conv = nn.Sequential(DWConv(c1, c1, k, stride, act=None), 
            SELayer(c1, reduction=se_ratio) if se_ratio else nn.Identity(),
            get_activation(act), ConvBnAct(c1, c2, 1, 1, act=None))
    else:
        c_ = round_channels(c1 * e, channel_divisor)
        self.conv = nn.Sequential(ConvBnAct(c1, c_, 1, 1, act=act), DWConv(
            c_, c_, k, stride, act=None), SELayer(c_, reduction=se_ratio) if
            se_ratio else nn.Identity(), get_activation(act), ConvBnAct(c_,
            c2, 1, 1, act=None))