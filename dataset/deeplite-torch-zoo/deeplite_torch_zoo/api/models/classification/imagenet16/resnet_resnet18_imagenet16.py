@MODEL_WRAPPER_REGISTRY.register(model_name='resnet18', dataset_name=
    'imagenet16', task_type='classification')
def resnet18_imagenet16(pretrained=False, num_classes=16):
    return _resnet_imagenet16('resnet18', pretrained=pretrained,
        num_classes=num_classes)
