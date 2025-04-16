def yolo_timm(model_name='yolo_timm_resnet18', dataset_name='voc',
    num_classes=20, pretrained=False, **kwargs):
    model = TimmYOLO(model_name.replace('yolo_timm_', ''), nc=num_classes,
        **kwargs)
    if pretrained:
        model = load_pretrained_model(model, model_name, dataset_name)
    return model
