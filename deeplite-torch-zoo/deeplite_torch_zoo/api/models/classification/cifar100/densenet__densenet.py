def _densenet(arch, block, layers, growth_rate=32, pretrained=False,
    num_classes=100):
    model = DenseNet(block, layers, growth_rate, num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url)
    return model
