def _gen_mobilenet_v3(variant, channel_multiplier=1.0, pretrained=False, **
    kwargs):
    """Creates a MobileNet-V3 large/small/minimal models.

    Ref impl: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v3.py
    Paper: https://arxiv.org/abs/1905.02244

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    if 'small' in variant:
        num_features = 1024
        if 'minimal' in variant:
            act_layer = 'relu'
            arch_def = [['ds_r1_k3_s2_e1_c16'], ['ir_r1_k3_s2_e4.5_c24',
                'ir_r1_k3_s1_e3.67_c24'], ['ir_r1_k3_s2_e4_c40',
                'ir_r2_k3_s1_e6_c40'], ['ir_r2_k3_s1_e3_c48'], [
                'ir_r3_k3_s2_e6_c96'], ['cn_r1_k1_s1_c576']]
        else:
            act_layer = 'hard_swish'
            arch_def = [['ds_r1_k3_s2_e1_c16_se0.25_nre'], [
                'ir_r1_k3_s2_e4.5_c24_nre', 'ir_r1_k3_s1_e3.67_c24_nre'], [
                'ir_r1_k5_s2_e4_c40_se0.25', 'ir_r2_k5_s1_e6_c40_se0.25'],
                ['ir_r2_k5_s1_e3_c48_se0.25'], ['ir_r3_k5_s2_e6_c96_se0.25'
                ], ['cn_r1_k1_s1_c576']]
    else:
        num_features = 1280
        if 'minimal' in variant:
            act_layer = 'relu'
            arch_def = [['ds_r1_k3_s1_e1_c16'], ['ir_r1_k3_s2_e4_c24',
                'ir_r1_k3_s1_e3_c24'], ['ir_r3_k3_s2_e3_c40'], [
                'ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80',
                'ir_r2_k3_s1_e2.3_c80'], ['ir_r2_k3_s1_e6_c112'], [
                'ir_r3_k3_s2_e6_c160'], ['cn_r1_k1_s1_c960']]
        else:
            act_layer = 'hard_swish'
            arch_def = [['ds_r1_k3_s1_e1_c16_nre'], [
                'ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'], [
                'ir_r3_k5_s2_e3_c40_se0.25_nre'], ['ir_r1_k3_s2_e6_c80',
                'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'], [
                'ir_r2_k3_s1_e6_c112_se0.25'], [
                'ir_r3_k5_s2_e6_c160_se0.25'], ['cn_r1_k1_s1_c960']]
    with layer_config_kwargs(kwargs):
        model_kwargs = dict(block_args=decode_arch_def(arch_def),
            num_features=num_features, stem_size=16, channel_multiplier=
            channel_multiplier, act_layer=resolve_act_layer(kwargs,
            act_layer), se_kwargs=dict(act_layer=get_act_layer('relu'),
            gate_fn=get_act_fn('hard_sigmoid'), reduce_mid=True, divisor=8),
            norm_kwargs=resolve_bn_args(kwargs), **kwargs)
        model = _create_model(model_kwargs, variant, pretrained)
    return model
