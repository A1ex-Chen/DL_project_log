def _mobilenetv3_vww(arch='small', pretrained=False, num_classes=2):
    if arch == 'small':
        model = mobilenetv3_small(num_classes=num_classes)
    elif arch == 'large':
        model = mobilenetv3_large(num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls[f'mobilenetv3_{arch}']
        model = load_pretrained_weights(model, checkpoint_url)
    return model
