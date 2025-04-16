def ResNet18(pretrained_on_imagenet=False, pretrained_dir=
    './trained_models/imagenet', **kwargs):
    if 'block' not in kwargs:
        kwargs['block'] = BasicBlock
    else:
        kwargs['block'] = eval(kwargs['block'])
    model = ResNet([2, 2, 2, 2], **kwargs)
    if 'input_channels' in kwargs and kwargs['input_channels'] == 1:
        input_channels = 1
    else:
        input_channels = 3
    if kwargs['block'] != BasicBlock and pretrained_on_imagenet:
        model = load_pretrained_with_different_encoder_block(model, kwargs[
            'block'].__name__, input_channels, 'r18', pretrained_dir=
            pretrained_dir)
    elif pretrained_on_imagenet:
        weights = model_zoo.load_url(model_urls['resnet18'], model_dir='./')
        if 'input_channels' in kwargs and kwargs['input_channels'] == 1:
            weights['conv1.weight'] = torch.sum(weights['conv1.weight'],
                axis=1, keepdim=True)
        weights.pop('fc.weight')
        weights.pop('fc.bias')
        model.load_state_dict(weights, strict=True)
        print('Loaded ResNet18 pretrained on ImageNet')
    return model
