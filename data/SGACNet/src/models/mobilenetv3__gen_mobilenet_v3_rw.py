def _gen_mobilenet_v3_rw(variant, channel_multiplier=1.0, pretrained=False,
    **kwargs):
    """Creates a MobileNet-V3 model (RW variant).

    Paper: https://arxiv.org/abs/1905.02244

    This was my first attempt at reproducing the MobileNet-V3 from paper alone. It came close to the
    eventual Tensorflow reference impl but has a few differences:
    1. This model has no bias on the head convolution
    2. This model forces no residual (noskip) on the first DWS block, this is different than MnasNet
    3. This model always uses ReLU for the SE activation layer, other models in the family inherit their act layer
       from their parent block
    4. This model does not enforce divisible by 8 limitation on the SE reduction channel count

    Overall the changes are fairly minor and result in a very small parameter count difference and no
    top-1/5

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [['ds_r1_k3_s1_e1_c16_nre_noskip'], [
        'ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'], [
        'ir_r3_k5_s2_e3_c40_se0.25_nre'], ['ir_r1_k3_s2_e6_c80',
        'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'], [
        'ir_r2_k3_s1_e6_c112_se0.25'], ['ir_r3_k5_s2_e6_c160_se0.25'], [
        'cn_r1_k1_s1_c960']]
    with layer_config_kwargs(kwargs):
        model_kwargs = dict(block_args=decode_arch_def(arch_def), head_bias
            =False, channel_multiplier=channel_multiplier, act_layer=
            resolve_act_layer(kwargs, 'hard_swish'), se_kwargs=dict(gate_fn
            =get_act_fn('hard_sigmoid'), reduce_mid=True), norm_kwargs=
            resolve_bn_args(kwargs), **kwargs)
        model = _create_model(model_kwargs, variant, pretrained)
    return model
