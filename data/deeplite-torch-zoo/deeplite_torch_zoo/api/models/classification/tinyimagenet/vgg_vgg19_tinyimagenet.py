@MODEL_WRAPPER_REGISTRY.register(model_name='vgg19', dataset_name=
    'tinyimagenet', task_type='classification')
def vgg19_tinyimagenet(pretrained=False, num_classes=100):
    model = models.vgg19(num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls['vgg19']
        model = load_pretrained_weights(model, checkpoint_url)
    return model
