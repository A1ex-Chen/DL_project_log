def __init__(self, in_channels, out_channels, kernel_size, stride=1,
    padding=0, dilation=1, groups=1, bias=True):
    super(Conv2dSame, self).__init__(in_channels, out_channels, kernel_size,
        stride, 0, dilation, groups, bias)
