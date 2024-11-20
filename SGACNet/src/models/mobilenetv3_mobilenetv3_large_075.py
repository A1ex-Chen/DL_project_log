def mobilenetv3_large_075(pretrained=False, **kwargs):
    """ MobileNet V3 Large 0.75"""
    model = _gen_mobilenet_v3('mobilenetv3_large_075', 0.75, pretrained=
        pretrained, **kwargs)
    return model
