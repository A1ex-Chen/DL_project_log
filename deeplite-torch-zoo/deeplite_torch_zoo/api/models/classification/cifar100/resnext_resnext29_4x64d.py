def resnext29_4x64d(pretrained=False):
    return _resnext('resnext29_4x64d', num_blocks=[3, 3, 3], cardinality=4,
        bottleneck_width=64, pretrained=pretrained)
