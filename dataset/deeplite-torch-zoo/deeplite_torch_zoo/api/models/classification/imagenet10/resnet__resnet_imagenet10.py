def _resnet_imagenet10(arch, pretrained=False, num_classes=10):
    model = torchvision.models.resnet18(num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url)
    return model
