def conv7x7(self, in_planes, out_planes, stride=1, groups=1, bn=False, act=
    False):
    """7x7 convolution with padding"""
    c = self.conv(7, in_planes, out_planes, groups=groups, stride=stride,
        bn=bn, act=act)
    return c
