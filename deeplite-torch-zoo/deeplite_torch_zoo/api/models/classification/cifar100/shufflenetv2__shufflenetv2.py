def _shufflenetv2(arch, net_size=1, pretrained=False, num_classes=100):
    model = ShuffleNetV2(net_size, num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url)
    return model
