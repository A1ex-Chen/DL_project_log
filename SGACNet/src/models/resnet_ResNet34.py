def ResNet34(pretrained_on_imagenet=False, pretrained_dir=
    './trained_models/imagenet', **kwargs):
    if 'block' not in kwargs:
        kwargs['block'] = BasicBlock
    elif kwargs['block'] in globals():
        kwargs['block'] = globals()[kwargs['block']]
    else:
        raise NotImplementedError('Block {} is not implemented'.format(
            kwargs['block']))
    if 'input_channels' in kwargs and kwargs['input_channels'] == 1:
        input_channels = 1
    else:
        input_channels = 3
    model = ResNet([3, 4, 6, 3], **kwargs)
    if kwargs['block'] != BasicBlock and pretrained_on_imagenet:
        model = load_pretrained_with_different_encoder_block(model, kwargs[
            'block'].__name__, input_channels, 'r34', pretrained_dir=
            pretrained_dir)
    elif pretrained_on_imagenet:
        weights = model_zoo.load_url(model_urls['resnet34'], model_dir='./')
        if input_channels == 1:
            weights['conv1.weight'] = torch.sum(weights['conv1.weight'],
                axis=1, keepdim=True)
        weights.pop('fc.weight')
        weights.pop('fc.bias')
        model.load_state_dict(weights, strict=True)
        print('Loaded ResNet34 pretrained on ImageNet')
    return model
