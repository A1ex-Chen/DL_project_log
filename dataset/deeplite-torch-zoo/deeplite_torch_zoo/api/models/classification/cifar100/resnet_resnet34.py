def resnet34(pretrained=False):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained=pretrained)
