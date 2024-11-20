def conv1x1T(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=
        stride, bias=False)
