def conv1x1(self, in_planes, out_planes, stride=1, groups=1, bn=False, act=
    False):
    """1x1 convolution with padding"""
    c = self.conv(1, in_planes, out_planes, groups=groups, stride=stride,
        bn=bn, act=act)
    return c
