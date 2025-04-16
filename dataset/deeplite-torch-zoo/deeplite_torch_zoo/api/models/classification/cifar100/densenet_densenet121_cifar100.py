@MODEL_WRAPPER_REGISTRY.register(model_name='densenet121', dataset_name=
    'cifar100', task_type='classification')
def densenet121_cifar100(pretrained=False, num_classes=100):
    return _densenet('densenet121', Bottleneck, [6, 12, 24, 16],
        growth_rate=32, num_classes=num_classes, pretrained=pretrained)
