@MODEL_WRAPPER_REGISTRY.register(model_name='resnext29_2x64d', dataset_name
    ='cifar100', task_type='classification')
def resnext29_2x64d_cifar100(pretrained=False, num_classes=100):
    return _resnext('resnext29_2x64d', num_classes=num_classes, num_blocks=
        [3, 3, 3], cardinality=2, bottleneck_width=64, pretrained=pretrained)
