@MODEL_WRAPPER_REGISTRY.register(model_name='lenet5', dataset_name='mnist',
    task_type='classification')
def lenet5_mnist(pretrained=False, num_classes=10):
    return _lenet_mnist('lenet5', pretrained=pretrained, num_classes=
        num_classes)
