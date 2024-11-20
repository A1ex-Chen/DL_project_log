def get_consistency_vae_config(block_out_channels=None, norm_num_groups=None):
    block_out_channels = block_out_channels or [2, 4]
    norm_num_groups = norm_num_groups or 2
    return {'encoder_block_out_channels': block_out_channels,
        'encoder_in_channels': 3, 'encoder_out_channels': 4,
        'encoder_down_block_types': ['DownEncoderBlock2D'] * len(
        block_out_channels), 'decoder_add_attention': False,
        'decoder_block_out_channels': block_out_channels,
        'decoder_down_block_types': ['ResnetDownsampleBlock2D'] * len(
        block_out_channels), 'decoder_downsample_padding': 1,
        'decoder_in_channels': 7, 'decoder_layers_per_block': 1,
        'decoder_norm_eps': 1e-05, 'decoder_norm_num_groups':
        norm_num_groups, 'encoder_norm_num_groups': norm_num_groups,
        'decoder_num_train_timesteps': 1024, 'decoder_out_channels': 6,
        'decoder_resnet_time_scale_shift': 'scale_shift',
        'decoder_time_embedding_type': 'learned', 'decoder_up_block_types':
        ['ResnetUpsampleBlock2D'] * len(block_out_channels),
        'scaling_factor': 1, 'latent_channels': 4}
