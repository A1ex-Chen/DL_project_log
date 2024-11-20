def mobilenetv3_rw(pretrained=False, **kwargs):
    """ MobileNet-V3 RW
    Attn: See note in gen function for this variant.
    """
    if pretrained:
        kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    model = _gen_mobilenet_v3_rw('mobilenetv3_rw', 1.0, pretrained=
        pretrained, **kwargs)
    return model
