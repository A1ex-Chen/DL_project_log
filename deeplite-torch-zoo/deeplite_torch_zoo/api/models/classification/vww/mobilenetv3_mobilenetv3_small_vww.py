@MODEL_WRAPPER_REGISTRY.register(model_name='mobilenetv3_small',
    dataset_name='vww', task_type='classification')
def mobilenetv3_small_vww(pretrained=False, num_classes=2):
    return _mobilenetv3_vww(arch='small', pretrained=pretrained,
        num_classes=num_classes)
