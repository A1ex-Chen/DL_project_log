def flexible_yolo(model_name='yolo_resnet18', dataset_name='voc',
    num_classes=20, pretrained=False, **kwargs):
    config_key = model_name
    config_path = get_project_root() / CFG_PATH / model_configs[config_key]
    backbone_kwargs, neck_kwargs = {}, {}
    if model_name in model_kwargs:
        backbone_kwargs, neck_kwargs = model_kwargs[model_name]['backbone'
            ], model_kwargs[model_name]['neck']
    model = FlexibleYOLO(str(config_path), nc=num_classes, backbone_kwargs=
        backbone_kwargs, neck_kwargs=neck_kwargs, **kwargs)
    if pretrained:
        model = load_pretrained_model(model, model_name, dataset_name)
    return model
