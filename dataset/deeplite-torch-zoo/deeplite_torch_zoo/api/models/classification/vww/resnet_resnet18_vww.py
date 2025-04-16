@MODEL_WRAPPER_REGISTRY.register(model_name='resnet18', dataset_name='vww',
    task_type='classification')
def resnet18_vww(pretrained=False, num_classes=2):
    return _resnet_vww('resnet18', pretrained=pretrained, num_classes=
        num_classes)
