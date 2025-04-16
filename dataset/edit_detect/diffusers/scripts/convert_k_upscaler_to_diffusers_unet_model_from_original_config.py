def unet_model_from_original_config(original_config):
    in_channels = original_config['input_channels'] + original_config[
        'unet_cond_dim']
    out_channels = original_config['input_channels'] + (1 if
        original_config['has_variance'] else 0)
    block_out_channels = original_config['channels']
    assert len(set(original_config['depths'])
        ) == 1, 'UNet2DConditionModel currently do not support blocks with different number of layers'
    layers_per_block = original_config['depths'][0]
    class_labels_dim = original_config['mapping_cond_dim']
    cross_attention_dim = original_config['cross_cond_dim']
    attn1_types = []
    attn2_types = []
    for s, c in zip(original_config['self_attn_depths'], original_config[
        'cross_attn_depths']):
        if s:
            a1 = 'self'
            a2 = 'cross' if c else None
        elif c:
            a1 = 'cross'
            a2 = None
        else:
            a1 = None
            a2 = None
        attn1_types.append(a1)
        attn2_types.append(a2)
    unet = UNet2DConditionModel(in_channels=in_channels, out_channels=
        out_channels, down_block_types=('KDownBlock2D',
        'KCrossAttnDownBlock2D', 'KCrossAttnDownBlock2D',
        'KCrossAttnDownBlock2D'), mid_block_type=None, up_block_types=(
        'KCrossAttnUpBlock2D', 'KCrossAttnUpBlock2D', 'KCrossAttnUpBlock2D',
        'KUpBlock2D'), block_out_channels=block_out_channels,
        layers_per_block=layers_per_block, act_fn='gelu', norm_num_groups=
        None, cross_attention_dim=cross_attention_dim, attention_head_dim=
        64, time_cond_proj_dim=class_labels_dim, resnet_time_scale_shift=
        'scale_shift', time_embedding_type='fourier', timestep_post_act=
        'gelu', conv_in_kernel=1, conv_out_kernel=1)
    return unet
