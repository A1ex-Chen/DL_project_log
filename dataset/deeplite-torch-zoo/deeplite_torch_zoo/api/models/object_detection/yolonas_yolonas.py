def yolonas(model_name='yolonas_s', dataset_name='voc', pretrained=False,
    num_classes=20, **kwargs):
    model = YOLONAS(arch_name=YOLONAS_CONFIGS[model_name], nc=num_classes,
        **kwargs)
    if pretrained:
        model = load_pretrained_model(model, model_name, dataset_name)
    return model
