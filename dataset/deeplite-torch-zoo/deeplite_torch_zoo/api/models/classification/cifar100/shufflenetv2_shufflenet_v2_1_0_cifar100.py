@MODEL_WRAPPER_REGISTRY.register(model_name='shufflenet_v2_1_0',
    dataset_name='cifar100', task_type='classification')
def shufflenet_v2_1_0_cifar100(pretrained=False, num_classes=100):
    return _shufflenetv2('shufflenet_v2', net_size=1, num_classes=
        num_classes, pretrained=pretrained)
