def pre_act_resnet152(pretrained=False):
    return _pre_act_resnet('pre_act_resnet152', PreActBottleneck, [3, 8, 36,
        3], pretrained=pretrained)
