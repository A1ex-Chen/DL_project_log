def conv3x3(self, in_planes, out_planes, stride=1, groups=1, bn=False, act=
    False):
    """3x3 convolution with padding"""
    c = self.conv(3, in_planes, out_planes, groups=groups, stride=stride,
        bn=bn, act=act)
    return c
