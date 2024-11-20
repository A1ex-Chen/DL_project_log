@MODEL_WRAPPER_REGISTRY.register(model_name='vgg19', dataset_name=
    'cifar100', task_type='classification')
def vgg19_cifar100(pretrained=False, num_classes=100):
    return _vgg('vgg19', pretrained, num_classes=num_classes)
