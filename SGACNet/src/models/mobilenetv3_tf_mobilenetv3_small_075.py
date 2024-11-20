def tf_mobilenetv3_small_075(pretrained=False, **kwargs):
    """ MobileNet V3 Small 0.75. Tensorflow compat variant. """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_075', 0.75, pretrained=
        pretrained, **kwargs)
    return model
