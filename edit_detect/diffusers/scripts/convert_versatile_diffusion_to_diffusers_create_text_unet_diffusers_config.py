def create_text_unet_diffusers_config(unet_params):
    """
    Creates a config for the diffusers based on the config of the VD model.
    """
    block_out_channels = [(unet_params.model_channels * mult) for mult in
        unet_params.channel_mult]
    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = 'CrossAttnDownBlockFlat' if unet_params.with_attn[i
            ] else 'DownBlockFlat'
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2
    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = 'CrossAttnUpBlockFlat' if unet_params.with_attn[-i - 1
            ] else 'UpBlockFlat'
        up_block_types.append(block_type)
        resolution //= 2
    if not all(n == unet_params.num_noattn_blocks[0] for n in unet_params.
        num_noattn_blocks):
        raise ValueError(
            'Not all num_res_blocks are equal, which is not supported in this script.'
            )
    config = {'sample_size': None, 'in_channels': (unet_params.
        input_channels, 1, 1), 'out_channels': (unet_params.output_channels,
        1, 1), 'down_block_types': tuple(down_block_types),
        'up_block_types': tuple(up_block_types), 'block_out_channels':
        tuple(block_out_channels), 'layers_per_block': unet_params.
        num_noattn_blocks[0], 'cross_attention_dim': unet_params.
        context_dim, 'attention_head_dim': unet_params.num_heads}
    return config
