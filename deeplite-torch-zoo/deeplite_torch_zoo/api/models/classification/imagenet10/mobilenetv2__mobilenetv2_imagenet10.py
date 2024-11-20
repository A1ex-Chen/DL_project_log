def _mobilenetv2_imagenet10(arch, alpha=1.0, pretrained=False, num_classes=10):
    model = torchvision.models.mobilenet.MobileNetV2(width_mult=alpha,
        num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url)
    return model
