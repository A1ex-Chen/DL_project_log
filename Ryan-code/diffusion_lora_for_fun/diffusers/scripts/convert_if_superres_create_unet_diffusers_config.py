def superres_create_unet_diffusers_config(original_unet_config):
    attention_resolutions = parse_list(original_unet_config[
        'attention_resolutions'])
    attention_resolutions = [(original_unet_config['image_size'] // int(res
        )) for res in attention_resolutions]
    channel_mult = parse_list(original_unet_config['channel_mult'])
    block_out_channels = [(original_unet_config['model_channels'] * mult) for
        mult in channel_mult]
    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        if resolution in attention_resolutions:
            block_type = 'SimpleCrossAttnDownBlock2D'
        elif original_unet_config['resblock_updown']:
            block_type = 'ResnetDownsampleBlock2D'
        else:
            block_type = 'DownBlock2D'
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2
    up_block_types = []
    for i in range(len(block_out_channels)):
        if resolution in attention_resolutions:
            block_type = 'SimpleCrossAttnUpBlock2D'
        elif original_unet_config['resblock_updown']:
            block_type = 'ResnetUpsampleBlock2D'
        else:
            block_type = 'UpBlock2D'
        up_block_types.append(block_type)
        resolution //= 2
    head_dim = original_unet_config['num_head_channels']
    use_linear_projection = original_unet_config['use_linear_in_transformer'
        ] if 'use_linear_in_transformer' in original_unet_config else False
    if use_linear_projection:
        if head_dim is None:
            head_dim = [5, 10, 20, 20]
    class_embed_type = None
    projection_class_embeddings_input_dim = None
    if 'num_classes' in original_unet_config:
        if original_unet_config['num_classes'] == 'sequential':
            class_embed_type = 'projection'
            assert 'adm_in_channels' in original_unet_config
            projection_class_embeddings_input_dim = original_unet_config[
                'adm_in_channels']
        else:
            raise NotImplementedError(
                f"Unknown conditional unet num_classes config: {original_unet_config['num_classes']}"
                )
    config = {'in_channels': original_unet_config['in_channels'],
        'down_block_types': tuple(down_block_types), 'block_out_channels':
        tuple(block_out_channels), 'layers_per_block': tuple(
        original_unet_config['num_res_blocks']), 'cross_attention_dim':
        original_unet_config['encoder_channels'], 'attention_head_dim':
        head_dim, 'use_linear_projection': use_linear_projection,
        'class_embed_type': class_embed_type,
        'projection_class_embeddings_input_dim':
        projection_class_embeddings_input_dim, 'out_channels':
        original_unet_config['out_channels'], 'up_block_types': tuple(
        up_block_types), 'upcast_attention': False, 'cross_attention_norm':
        'group_norm', 'mid_block_type': 'UNetMidBlock2DSimpleCrossAttn',
        'act_fn': 'gelu'}
    if original_unet_config['use_scale_shift_norm']:
        config['resnet_time_scale_shift'] = 'scale_shift'
    return config
