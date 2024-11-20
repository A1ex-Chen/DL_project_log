def yolov6(model_name='yolo6n', config_path=None, dataset_name='voc',
    pretrained=False, num_classes=20, **kwargs):
    model = YOLOv6(model_config=config_path, nc=num_classes, is_lite='lite' in
        model_name, **kwargs)
    if pretrained:
        model = load_pretrained_model(model, model_name, dataset_name)
    return model
