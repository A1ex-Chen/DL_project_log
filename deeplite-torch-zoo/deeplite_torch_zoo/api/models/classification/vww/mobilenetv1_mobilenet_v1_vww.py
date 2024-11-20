@MODEL_WRAPPER_REGISTRY.register(model_name='mobilenet_v1', dataset_name=
    'vww', task_type='classification')
def mobilenet_v1_vww(pretrained=False, num_classes=2):
    return mobilenetv1_vww(model_name='mobilenet_v1', pretrained=pretrained,
        num_classes=num_classes)
