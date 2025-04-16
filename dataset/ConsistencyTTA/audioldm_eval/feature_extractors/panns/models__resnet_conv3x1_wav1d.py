def _resnet_conv3x1_wav1d(in_planes, out_planes, dilation):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=1,
        padding=dilation, groups=1, bias=False, dilation=dilation)
