def resnet(pretrained=False, **kwargs):
    VERSION_FN_MAP = {'18': resnet18, '34': resnet34, '50': resnet50, '101':
        resnet101, '152': resnet152}
    version = str(kwargs.pop('version'))
    return VERSION_FN_MAP[version](pretrained, **kwargs)
