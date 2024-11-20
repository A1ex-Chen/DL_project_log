def _mobilenetv2(arch, pretrained=False, num_classes=100):
    model = MobileNetV2(num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url)
    return model
