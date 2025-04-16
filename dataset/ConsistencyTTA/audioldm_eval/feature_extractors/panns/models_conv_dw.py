def conv_dw(inp, oup, stride):
    _layers = [nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False), nn.
        AvgPool2d(stride), nn.BatchNorm2d(inp), nn.ReLU(inplace=True), nn.
        Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU
        (inplace=True)]
    _layers = nn.Sequential(*_layers)
    init_layer(_layers[0])
    init_bn(_layers[2])
    init_layer(_layers[4])
    init_bn(_layers[5])
    return _layers
