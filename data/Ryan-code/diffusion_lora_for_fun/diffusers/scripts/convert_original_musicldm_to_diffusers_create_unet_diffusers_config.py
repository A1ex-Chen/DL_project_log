def create_unet_diffusers_config(original_config, image_size: int):
    """
    Creates a UNet config for diffusers based on the config of the original MusicLDM model.
    """
    unet_params = original_config['model']['params']['unet_config']['params']
    vae_params = original_config['model']['params']['first_stage_config'][
        'params']['ddconfig']
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
    cross_attention_dim = unet_params['cross_attention_dim'
        ] if 'cross_attention_dim' in unet_params else block_out_channels
    class_embed_type = ('simple_projection' if 'extra_film_condition_dim' in
        unet_params else None)
    projection_class_embeddings_input_dim = unet_params[
        'extra_film_condition_dim'
        ] if 'extra_film_condition_dim' in unet_params else None
    class_embeddings_concat = unet_params['extra_film_use_concat'
        ] if 'extra_film_use_concat' in unet_params else None
    config = {'sample_size': image_size // vae_scale_factor, 'in_channels':
        unet_params['in_channels'], 'out_channels': unet_params[
        'out_channels'], 'down_block_types': tuple(down_block_types),
        'up_block_types': tuple(up_block_types), 'block_out_channels':
        tuple(block_out_channels), 'layers_per_block': unet_params[
        'num_res_blocks'], 'cross_attention_dim': cross_attention_dim,
        'class_embed_type': class_embed_type,
        'projection_class_embeddings_input_dim':
        projection_class_embeddings_input_dim, 'class_embeddings_concat':
        class_embeddings_concat}
    return config
