@MODEL_WRAPPER_REGISTRY.register(model_name='mlp4', dataset_name='mnist',
    task_type='classification')
def mlp4_mnist(pretrained=False, num_classes=10):
    return _mlp10_mnist('mlp4', n_hiddens=[128, 128, 128, 128], pretrained=
        pretrained, num_classes=num_classes)
