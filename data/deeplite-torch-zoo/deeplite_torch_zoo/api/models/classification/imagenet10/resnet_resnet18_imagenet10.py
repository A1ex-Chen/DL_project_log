@MODEL_WRAPPER_REGISTRY.register(model_name='resnet18', dataset_name=
    'imagenet10', task_type='classification')
def resnet18_imagenet10(pretrained=False, num_classes=10):
    return _resnet_imagenet10('resnet18', pretrained=pretrained,
        num_classes=num_classes)
