def create_unet_config(original_config, image_size: int, attention_type):
    unet_params = original_config['model']['params']
    vae_params = original_config['autoencoder']['params']['ddconfig']
    block_out_channels = [(unet_params['model_channels'] * mult) for mult in
        unet_params['channel_mult']]
    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = 'CrossAttnDownBlock2D' if resolution in unet_params[
            'attention_resolutions'] else 'DownBlock2D'
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2
    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = 'CrossAttnUpBlock2D' if resolution in unet_params[
            'attention_resolutions'] else 'UpBlock2D'
        up_block_types.append(block_type)
        resolution //= 2
    vae_scale_factor = 2 ** (len(vae_params['ch_mult']) - 1)
    head_dim = unet_params['num_heads'] if 'num_heads' in unet_params else None
    use_linear_projection = unet_params['use_linear_in_transformer'
        ] if 'use_linear_in_transformer' in unet_params else False
    if use_linear_projection:
        if head_dim is None:
            head_dim = [5, 10, 20, 20]
    config = {'sample_size': image_size // vae_scale_factor, 'in_channels':
        unet_params['in_channels'], 'down_block_types': tuple(
        down_block_types), 'block_out_channels': tuple(block_out_channels),
        'layers_per_block': unet_params['num_res_blocks'],
        'cross_attention_dim': unet_params['context_dim'],
        'attention_head_dim': head_dim, 'use_linear_projection':
        use_linear_projection, 'attention_type': attention_type}
    return config
