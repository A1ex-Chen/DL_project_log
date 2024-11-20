def mobilenetv1_vww(model_name, num_classes=2, last_pooling_size=7,
    width_mult=1.0, pretrained=False):
    model = MobileNetV1(num_classes=num_classes, width_mult=width_mult,
        last_pooling_size=last_pooling_size)
    if pretrained:
        checkpoint_url = model_urls[model_name]
        model = load_pretrained_weights(model, checkpoint_url)
    return model
