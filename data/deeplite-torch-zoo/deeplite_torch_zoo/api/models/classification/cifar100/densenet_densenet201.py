def densenet201(pretrained=False):
    return _densenet('densenet201', Bottleneck, [6, 12, 48, 32],
        growth_rate=32, pretrained=pretrained)
