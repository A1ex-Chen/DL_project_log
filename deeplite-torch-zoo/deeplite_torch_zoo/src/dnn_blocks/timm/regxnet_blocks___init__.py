def __init__(self, c1, c2, stride=1, exp_ratio=1.0, k=3, se_ratio=None,
    ch_div=1, act='hswish', dw_act='relu6'):
    super(RexNetBottleneck, self).__init__()
    self.in_channels = c1
    self.out_channels = c2
    if exp_ratio != 1.0:
        dw_chs = round_channels(round(c1 * exp_ratio), divisor=ch_div)
        self.conv_exp = ConvBnAct(c1, dw_chs, act=act)
    else:
        dw_chs = c1
        self.conv_exp = None
    self.conv_dw = ConvBnAct(dw_chs, dw_chs, k, s=stride, g=dw_chs, act=None)
    self.se = SEWithNorm(dw_chs, mid_channels=round_channels(int(dw_chs *
        se_ratio), ch_div)) if se_ratio is not None else nn.Identity()
    self.act_dw = get_activation(dw_act)
    self.conv_pwl = ConvBnAct(dw_chs, c2, 1, act=None)
