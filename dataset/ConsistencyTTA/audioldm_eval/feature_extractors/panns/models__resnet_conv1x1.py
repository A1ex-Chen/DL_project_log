def _resnet_conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False
        )
