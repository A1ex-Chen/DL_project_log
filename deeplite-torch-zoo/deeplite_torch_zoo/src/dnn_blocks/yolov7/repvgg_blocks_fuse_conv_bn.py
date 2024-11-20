def fuse_conv_bn(self, conv, bn):
    std = (bn.running_var + bn.eps).sqrt()
    bias = bn.bias - bn.running_mean * bn.weight / std
    t = (bn.weight / std).reshape(-1, 1, 1, 1)
    weights = conv.weight * t
    bn = nn.Identity()
    conv = nn.Conv2d(in_channels=conv.in_channels, out_channels=conv.
        out_channels, kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, dilation=conv.dilation, groups=conv.groups,
        bias=True, padding_mode=conv.padding_mode)
    conv.weight = torch.nn.Parameter(weights)
    conv.bias = torch.nn.Parameter(bias)
    return conv
