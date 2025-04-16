def _resnext(arch, num_blocks, cardinality, bottleneck_width, num_classes=
    100, pretrained=False):
    model = ResNeXt(num_blocks=num_blocks, cardinality=cardinality,
        bottleneck_width=bottleneck_width, num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url)
    return model
