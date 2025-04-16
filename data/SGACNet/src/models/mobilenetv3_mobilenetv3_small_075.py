def mobilenetv3_small_075(pretrained=False, **kwargs):
    """ MobileNet V3 Small 0.75 """
    model = _gen_mobilenet_v3('mobilenetv3_small_075', 0.75, pretrained=
        pretrained, **kwargs)
    return model
