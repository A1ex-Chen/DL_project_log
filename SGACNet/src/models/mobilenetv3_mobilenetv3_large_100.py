def mobilenetv3_large_100(pretrained=False, **kwargs):
    """ MobileNet V3 Large 1.0 """
    model = _gen_mobilenet_v3('mobilenetv3_large_100', 1.0, pretrained=
        pretrained, **kwargs)
    return model
