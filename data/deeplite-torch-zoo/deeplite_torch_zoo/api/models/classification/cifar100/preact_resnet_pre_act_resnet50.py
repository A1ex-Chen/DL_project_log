def pre_act_resnet50(pretrained=False):
    return _pre_act_resnet('pre_act_resnet50', PreActBottleneck, [3, 4, 6, 
        3], pretrained=pretrained)
