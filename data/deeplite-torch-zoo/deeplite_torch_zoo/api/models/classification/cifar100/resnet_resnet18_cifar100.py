@MODEL_WRAPPER_REGISTRY.register(model_name='resnet18', dataset_name=
    'cifar100', task_type='classification')
def resnet18_cifar100(pretrained=False, num_classes=100):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], num_classes=
        num_classes, pretrained=pretrained)
