@MODEL_WRAPPER_REGISTRY.register(model_name='mobilenet_v1', dataset_name=
    'cifar100', task_type='classification')
def mobilenet_v1_cifar100(pretrained=False, num_classes=100):
    return _mobilenetv1('mobilenet_v1', pretrained, num_classes=num_classes)
