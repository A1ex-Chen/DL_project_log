def _build_net(self) ->nn.Module:
    if self.pretrained:
        efficientnet_b7 = EfficientNet.from_pretrained(model_name=
            'efficientnet-b7')
    else:
        efficientnet_b7 = EfficientNet.from_name(model_name='efficientnet-b7')
    efficientnet_b7._fc = nn.Linear(in_features=efficientnet_b7._fc.
        in_features, out_features=self.num_classes)
    conv1 = nn.ModuleList([efficientnet_b7._conv_stem, efficientnet_b7._bn0,
        efficientnet_b7._swish] + list(efficientnet_b7._blocks[:4]))
    conv2 = efficientnet_b7._blocks[4:11]
    conv3 = efficientnet_b7._blocks[11:18]
    conv4 = efficientnet_b7._blocks[18:38]
    conv5 = nn.ModuleList(list(efficientnet_b7._blocks[38:]) + [
        efficientnet_b7._conv_head, efficientnet_b7._bn1, efficientnet_b7.
        _swish])
    modules = [conv1, conv2, conv3, conv4, conv5]
    assert 0 <= self.num_frozen_levels <= len(modules)
    freezing_modules = modules[:self.num_frozen_levels]
    for module in freezing_modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False
    return efficientnet_b7
