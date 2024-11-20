def densenet161(pretrained=False):
    return _densenet('densenet161', Bottleneck, [6, 12, 36, 24],
        growth_rate=48, pretrained=pretrained)
