@MODEL_WRAPPER_REGISTRY.register(model_name='mobilenet_v2_0_35',
    dataset_name='imagenet10', task_type='classification')
def mobilenet_v2_0_35_imagenet10(pretrained=False, num_classes=10):
    return _mobilenetv2_imagenet10('mobilenetv2_0.35', alpha=0.35,
        pretrained=pretrained, num_classes=num_classes)
