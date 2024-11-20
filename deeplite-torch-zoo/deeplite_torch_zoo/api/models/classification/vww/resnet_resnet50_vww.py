@MODEL_WRAPPER_REGISTRY.register(model_name='resnet50', dataset_name='vww',
    task_type='classification')
def resnet50_vww(pretrained=False, num_classes=2):
    return _resnet_vww('resnet50', pretrained=pretrained, num_classes=
        num_classes)
