@MODEL_WRAPPER_REGISTRY.register(model_name='mobilenetv3_large',
    dataset_name='vww', task_type='classification')
def mobilenetv3_large_vww(pretrained=False, num_classes=2):
    return _mobilenetv3_vww(arch='large', pretrained=pretrained,
        num_classes=num_classes)
