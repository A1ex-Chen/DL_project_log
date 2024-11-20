@MODEL_WRAPPER_REGISTRY.register(model_name='googlenet', dataset_name=
    'cifar100', task_type='classification')
def googlenet_cifar100(pretrained=False, num_classes=100):
    return _googlenet('googlenet', pretrained, num_classes=num_classes)
