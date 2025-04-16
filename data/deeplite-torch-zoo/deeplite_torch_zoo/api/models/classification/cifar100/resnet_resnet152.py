def resnet152(pretrained=False):
    return _resnet('resnet34', Bottleneck, [3, 8, 36, 3], pretrained=pretrained
        )
