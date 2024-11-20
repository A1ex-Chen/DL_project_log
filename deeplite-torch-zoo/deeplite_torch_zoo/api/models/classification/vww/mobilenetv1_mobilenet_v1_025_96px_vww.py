@MODEL_WRAPPER_REGISTRY.register(model_name='mobilenet_v1_0.25_96px',
    dataset_name='vww', task_type='classification')
def mobilenet_v1_025_96px_vww(pretrained=False, num_classes=2, width_mult=
    0.25, last_pooling_size=3):
    return mobilenetv1_vww(model_name='mobilenet_v1_0.25_96px', pretrained=
        pretrained, num_classes=num_classes, width_mult=width_mult,
        last_pooling_size=last_pooling_size)
