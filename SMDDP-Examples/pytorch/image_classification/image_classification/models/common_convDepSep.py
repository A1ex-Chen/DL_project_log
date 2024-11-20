def convDepSep(self, kernel_size, in_planes, out_planes, stride=1, bn=False,
    act=False):
    """3x3 depthwise separable convolution with padding"""
    c = self.conv(kernel_size, in_planes, out_planes, groups=in_planes,
        stride=stride, bn=bn, act=act)
    return c
