def conv_bn(inp, oup, stride):
    _layers = [nn.Conv2d(inp, oup, 3, 1, 1, bias=False), nn.AvgPool2d(
        stride), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True)]
    _layers = nn.Sequential(*_layers)
    init_layer(_layers[0])
    init_bn(_layers[2])
    return _layers
