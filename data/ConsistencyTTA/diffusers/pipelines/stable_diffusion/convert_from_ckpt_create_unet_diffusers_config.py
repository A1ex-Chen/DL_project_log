def create_unet_diffusers_config(original_config, image_size: int,
    controlnet=False):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    if controlnet:
        unet_params = original_config.model.params.control_stage_config.params
    else:
        unet_params = original_config.model.params.unet_config.params
    vae_params = (original_config.model.params.first_stage_config.params.
        ddconfig)
    block_out_channels = [(unet_params.model_channels * mult) for mult in
        unet_params.channel_mult]
    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = ('CrossAttnDownBlock2D' if resolution in unet_params.
            attention_resolutions else 'DownBlock2D')
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2
    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = ('CrossAttnUpBlock2D' if resolution in unet_params.
            attention_resolutions else 'UpBlock2D')
        up_block_types.append(block_type)
        resolution //= 2
    vae_scale_factor = 2 ** (len(vae_params.ch_mult) - 1)
    head_dim = unet_params.num_heads if 'num_heads' in unet_params else None
    use_linear_projection = (unet_params.use_linear_in_transformer if 
        'use_linear_in_transformer' in unet_params else False)
    if use_linear_projection:
        if head_dim is None:
            head_dim = [5, 10, 20, 20]
    class_embed_type = None
    projection_class_embeddings_input_dim = None
    if 'num_classes' in unet_params:
        if unet_params.num_classes == 'sequential':
            class_embed_type = 'projection'
            assert 'adm_in_channels' in unet_params
            projection_class_embeddings_input_dim = unet_params.adm_in_channels
        else:
            raise NotImplementedError(
                f'Unknown conditional unet num_classes config: {unet_params.num_classes}'
                )
    config = {'sample_size': image_size // vae_scale_factor, 'in_channels':
        unet_params.in_channels, 'down_block_types': tuple(down_block_types
        ), 'block_out_channels': tuple(block_out_channels),
        'layers_per_block': unet_params.num_res_blocks,
        'cross_attention_dim': unet_params.context_dim,
        'attention_head_dim': head_dim, 'use_linear_projection':
        use_linear_projection, 'class_embed_type': class_embed_type,
        'projection_class_embeddings_input_dim':
        projection_class_embeddings_input_dim}
    if not controlnet:
        config['out_channels'] = unet_params.out_channels
        config['up_block_types'] = tuple(up_block_types)
    return config
