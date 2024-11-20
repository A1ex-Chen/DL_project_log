def _resnet(arch, block, layers, **kwargs):
    return ResNet(block, layers, **kwargs)
