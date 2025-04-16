@MODEL_WRAPPER_REGISTRY.register(model_name='pre_act_resnet18',
    dataset_name='cifar100', task_type='classification')
def pre_act_resnet18_cifar100(pretrained=False, num_classes=100):
    return _pre_act_resnet('pre_act_resnet18', PreActBlock, [2, 2, 2, 2],
        num_classes=num_classes, pretrained=pretrained)
