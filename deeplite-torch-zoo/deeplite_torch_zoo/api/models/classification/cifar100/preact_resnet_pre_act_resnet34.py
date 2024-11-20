def pre_act_resnet34(pretrained=False):
    return _pre_act_resnet('pre_act_resnet34', PreActBlock, [3, 4, 6, 3],
        pretrained=pretrained)
