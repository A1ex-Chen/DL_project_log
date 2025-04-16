def resnet101(pretrained=False):
    return _resnet('resnet34', Bottleneck, [3, 4, 23, 3], pretrained=pretrained
        )
