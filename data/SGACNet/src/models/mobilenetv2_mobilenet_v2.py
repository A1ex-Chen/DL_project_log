def mobilenet_v2(multiplier=1.0, pretrained_on_imagenet=False, progress=
    True, pretrained_dir='./trained_models/imagenet', **kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained_on_imagenet:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
            progress=progress)
        model.load_state_dict(state_dict)
        print('Loaded mobilenet_v2 pretrained on ImageNet')
    return model
