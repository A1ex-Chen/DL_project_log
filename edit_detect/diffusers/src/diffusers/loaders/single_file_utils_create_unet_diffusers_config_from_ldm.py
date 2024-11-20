def create_unet_diffusers_config_from_ldm(original_config, checkpoint,
    image_size=None, upcast_attention=None, num_in_channels=None):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    if image_size is not None:
        deprecation_message = (
            'Configuring UNet2DConditionModel with the `image_size` argument to `from_single_file`is deprecated and will be ignored in future versions.'
            )
        deprecate('image_size', '1.0.0', deprecation_message)
    image_size = set_image_size(checkpoint, image_size=image_size)
    if 'unet_config' in original_config['model']['params'] and original_config[
        'model']['params']['unet_config'] is not None:
        unet_params = original_config['model']['params']['unet_config'][
            'params']
    else:
        unet_params = original_config['model']['params']['network_config'][
            'params']
    if num_in_channels is not None:
        deprecation_message = (
            'Configuring UNet2DConditionModel with the `num_in_channels` argument to `from_single_file`is deprecated and will be ignored in future versions.'
            )
        deprecate('image_size', '1.0.0', deprecation_message)
        in_channels = num_in_channels
    else:
        in_channels = unet_params['in_channels']
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
    if unet_params['transformer_depth'] is not None:
        transformer_layers_per_block = unet_params['transformer_depth'
            ] if isinstance(unet_params['transformer_depth'], int) else list(
            unet_params['transformer_depth'])
    else:
        transformer_layers_per_block = 1
    vae_scale_factor = 2 ** (len(vae_params['ch_mult']) - 1)
    head_dim = unet_params['num_heads'] if 'num_heads' in unet_params else None
    use_linear_projection = unet_params['use_linear_in_transformer'
        ] if 'use_linear_in_transformer' in unet_params else False
    if use_linear_projection:
        if head_dim is None:
            head_dim_mult = unet_params['model_channels'] // unet_params[
                'num_head_channels']
            head_dim = [(head_dim_mult * c) for c in list(unet_params[
                'channel_mult'])]
    class_embed_type = None
    addition_embed_type = None
    addition_time_embed_dim = None
    projection_class_embeddings_input_dim = None
    context_dim = None
    if unet_params['context_dim'] is not None:
        context_dim = unet_params['context_dim'] if isinstance(unet_params[
            'context_dim'], int) else unet_params['context_dim'][0]
    if 'num_classes' in unet_params:
        if unet_params['num_classes'] == 'sequential':
            if context_dim in [2048, 1280]:
                addition_embed_type = 'text_time'
                addition_time_embed_dim = 256
            else:
                class_embed_type = 'projection'
            assert 'adm_in_channels' in unet_params
            projection_class_embeddings_input_dim = unet_params[
                'adm_in_channels']
    config = {'sample_size': image_size // vae_scale_factor, 'in_channels':
        in_channels, 'down_block_types': down_block_types,
        'block_out_channels': block_out_channels, 'layers_per_block':
        unet_params['num_res_blocks'], 'cross_attention_dim': context_dim,
        'attention_head_dim': head_dim, 'use_linear_projection':
        use_linear_projection, 'class_embed_type': class_embed_type,
        'addition_embed_type': addition_embed_type,
        'addition_time_embed_dim': addition_time_embed_dim,
        'projection_class_embeddings_input_dim':
        projection_class_embeddings_input_dim,
        'transformer_layers_per_block': transformer_layers_per_block}
    if upcast_attention is not None:
        deprecation_message = (
            'Configuring UNet2DConditionModel with the `upcast_attention` argument to `from_single_file`is deprecated and will be ignored in future versions.'
            )
        deprecate('image_size', '1.0.0', deprecation_message)
        config['upcast_attention'] = upcast_attention
    if 'disable_self_attentions' in unet_params:
        config['only_cross_attention'] = unet_params['disable_self_attentions']
    if 'num_classes' in unet_params and isinstance(unet_params[
        'num_classes'], int):
        config['num_class_embeds'] = unet_params['num_classes']
    config['out_channels'] = unet_params['out_channels']
    config['up_block_types'] = up_block_types
    return config
