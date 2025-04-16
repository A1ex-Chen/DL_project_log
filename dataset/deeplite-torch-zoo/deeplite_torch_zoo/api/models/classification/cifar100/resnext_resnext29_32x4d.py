def resnext29_32x4d(pretrained=False):
    return _resnext('resnext29_32x4d', num_blocks=[3, 3, 3], cardinality=32,
        bottleneck_width=4, pretrained=pretrained)
