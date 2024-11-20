@staticmethod
def _make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers.append(nn.Conv2d(in_channels, x, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(x))
            layers.append(nn.ReLU(inplace=True))
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return layers
