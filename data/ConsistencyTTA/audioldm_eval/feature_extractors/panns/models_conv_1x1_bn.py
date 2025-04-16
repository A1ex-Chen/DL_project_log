def conv_1x1_bn(inp, oup):
    _layers = nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.
        BatchNorm2d(oup), nn.ReLU6(inplace=True))
    init_layer(_layers[0])
    init_bn(_layers[1])
    return _layers
