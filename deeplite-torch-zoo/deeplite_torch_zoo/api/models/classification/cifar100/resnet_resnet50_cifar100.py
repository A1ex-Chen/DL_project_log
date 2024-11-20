@MODEL_WRAPPER_REGISTRY.register(model_name='resnet50', dataset_name=
    'cifar100', task_type='classification')
def resnet50_cifar100(pretrained=False, num_classes=100):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], num_classes=
        num_classes, pretrained=pretrained)
