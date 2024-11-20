def ResNet50(pretrained_on_imagenet=False, **kwargs):
    model = ResNet([3, 4, 6, 3], Bottleneck, **kwargs)
    if 'input_channels' in kwargs and kwargs['input_channels'] == 1:
        input_channels = 1
    else:
        input_channels = 3
    if pretrained_on_imagenet:
        weights = model_zoo.load_url(model_urls['resnet50'], model_dir='./')
        if input_channels == 1:
            weights['conv1.weight'] = torch.sum(weights['conv1.weight'],
                axis=1, keepdim=True)
        weights.pop('fc.weight')
        weights.pop('fc.bias')
        model.load_state_dict(weights, strict=True)
        print('Loaded ResNet50 pretrained on ImageNet')
    return model
