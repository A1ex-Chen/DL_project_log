def _resnet_imagenet16(arch, pretrained=False, num_classes=16):
    if arch == 'resnet18':
        model = torchvision.models.resnet18(num_classes=num_classes)
    elif arch == 'resnet50':
        model = torchvision.models.resnet50(num_classes=num_classes)
    else:
        raise ValueError
    if pretrained:
        checkpoint_url = model_urls[arch]
        model = load_pretrained_weights(model, checkpoint_url)
    return model
