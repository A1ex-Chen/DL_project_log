def __init__(self, c1, c2, k=7, s=4, p=None, g=1, act=True,
    layer_scale_init_value=1e-06):
    super(RobustConv2, self).__init__()
    self.conv_strided = ConvBnAct(c1, c1, k=k, s=s, p=p, g=c1, act=act)
    self.conv_deconv = nn.ConvTranspose2d(in_channels=c1, out_channels=c2,
        kernel_size=s, stride=s, padding=0, bias=True, dilation=1, groups=1)
    self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(c2)
        ) if layer_scale_init_value > 0 else None
