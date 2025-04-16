def resnext29_8x64d(pretrained=False):
    return _resnext('resnext29_8x64d', num_blocks=[3, 3, 3], cardinality=8,
        bottleneck_width=64, pretrained=pretrained)
