def create_yolo_model(model_name='yolo5s', dataset_name='voc', num_classes=
    20, config_path=None, pretrained=False, **kwargs):
    model = YOLO(config_path, nc=num_classes, **kwargs)
    if pretrained:
        model = load_pretrained_model(model, model_name, dataset_name)
    return model
