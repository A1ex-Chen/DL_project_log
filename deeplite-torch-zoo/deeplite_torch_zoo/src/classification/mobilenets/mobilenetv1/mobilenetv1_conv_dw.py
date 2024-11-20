def conv_dw(inp, oup, stride):
    inp, oup = int(inp * width_mult), int(oup * width_mult)
    return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias
        =False), nn.BatchNorm2d(inp), nn.ReLU(inplace=True), nn.Conv2d(inp,
        oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))
