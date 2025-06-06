def _resnet(arch, block, layers, num_classes=100, pretrained=False):
    model = ResNet(block, layers, num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url)
    return model
