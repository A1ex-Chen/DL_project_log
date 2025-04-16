def _resnet_conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
        padding=1, groups=1, bias=False, dilation=1)
