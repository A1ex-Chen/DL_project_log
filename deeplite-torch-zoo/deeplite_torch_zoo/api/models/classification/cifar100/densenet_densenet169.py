def densenet169(pretrained=False):
    return _densenet('densenet169', Bottleneck, [6, 12, 32, 32],
        growth_rate=32, pretrained=pretrained)
