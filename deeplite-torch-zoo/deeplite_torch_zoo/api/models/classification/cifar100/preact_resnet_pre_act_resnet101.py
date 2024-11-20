def pre_act_resnet101(pretrained=False):
    return _pre_act_resnet('pre_act_resnet101', PreActBottleneck, [3, 4, 23,
        3], pretrained=pretrained)
