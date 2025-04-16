def conv_bn(inp, oup, stride):
    oup = int(oup * width_mult)
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.
        BatchNorm2d(oup), nn.ReLU(inplace=True))
