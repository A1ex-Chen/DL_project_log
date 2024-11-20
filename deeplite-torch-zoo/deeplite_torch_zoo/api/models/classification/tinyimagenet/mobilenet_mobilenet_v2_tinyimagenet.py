@MODEL_WRAPPER_REGISTRY.register(model_name='mobilenet_v2', dataset_name=
    'tinyimagenet', task_type='classification')
def mobilenet_v2_tinyimagenet(pretrained=False, num_classes=100):
    model = models.mobilenet_v2(num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls['mobilenet_v2']
        model = load_pretrained_weights(model, checkpoint_url)
    return model
