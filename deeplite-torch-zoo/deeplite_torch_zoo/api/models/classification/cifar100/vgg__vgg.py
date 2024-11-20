def _vgg(cfg, pretrained=False, num_classes=100):
    model = VGG(cfg, num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls[cfg]
        model = load_pretrained_weights(model, checkpoint_url)
    return model
