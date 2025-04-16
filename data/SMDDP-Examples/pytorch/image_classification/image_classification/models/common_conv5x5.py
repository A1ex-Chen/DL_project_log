def conv5x5(self, in_planes, out_planes, stride=1, groups=1, bn=False, act=
    False):
    """5x5 convolution with padding"""
    c = self.conv(5, in_planes, out_planes, groups=groups, stride=stride,
        bn=bn, act=act)
    return c
