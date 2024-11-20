@MODEL_WRAPPER_REGISTRY.register(model_name='resnet50', dataset_name=
    'tinyimagenet', task_type='classification')
def resnet50_tinyimagenet(pretrained=False, num_classes=100):
    model = models.resnet50(num_classes=num_classes)
    if pretrained:
        checkpoint_url = model_urls['resnet50']
        model = load_pretrained_weights(model, checkpoint_url)
    return model
