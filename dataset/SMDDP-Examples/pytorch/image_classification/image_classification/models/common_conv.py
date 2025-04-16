def conv(self, kernel_size, in_planes, out_planes, groups=1, stride=1, bn=
    False, zero_init_bn=False, act=False):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, groups
        =groups, stride=stride, padding=int((kernel_size - 1) / 2), bias=False)
    nn.init.kaiming_normal_(conv.weight, mode=self.config.conv_init,
        nonlinearity='relu')
    layers = [('conv', conv)]
    if bn:
        layers.append(('bn', self.batchnorm(out_planes, zero_init_bn)))
    if act:
        layers.append(('act', self.activation()))
    if bn or act:
        return nn.Sequential(OrderedDict(layers))
    else:
        return conv
