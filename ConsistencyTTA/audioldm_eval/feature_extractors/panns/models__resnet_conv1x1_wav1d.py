def _resnet_conv1x1_wav1d(in_planes, out_planes):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, bias=False
        )
