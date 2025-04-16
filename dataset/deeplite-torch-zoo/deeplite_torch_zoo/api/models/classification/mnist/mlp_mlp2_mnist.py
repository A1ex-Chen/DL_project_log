@MODEL_WRAPPER_REGISTRY.register(model_name='mlp2', dataset_name='mnist',
    task_type='classification')
def mlp2_mnist(pretrained=False, num_classes=10):
    return _mlp10_mnist('mlp2', n_hiddens=[128, 128], pretrained=pretrained,
        num_classes=num_classes)
