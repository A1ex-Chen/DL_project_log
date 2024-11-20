@MODEL_WRAPPER_REGISTRY.register(model_name='mobilenet_v2', dataset_name=
    'cifar100', task_type='classification')
def mobilenet_v2_cifar100(pretrained=False, num_classes=100):
    return _mobilenetv2('mobilenet_v2', pretrained, num_classes=num_classes)
