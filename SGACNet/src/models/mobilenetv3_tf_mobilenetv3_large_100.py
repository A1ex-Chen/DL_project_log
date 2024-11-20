def tf_mobilenetv3_large_100(pretrained=False, **kwargs):
    """ MobileNet V3 Large 1.0. Tensorflow compat variant. """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_100', 1.0, pretrained=
        pretrained, **kwargs)
    return model
