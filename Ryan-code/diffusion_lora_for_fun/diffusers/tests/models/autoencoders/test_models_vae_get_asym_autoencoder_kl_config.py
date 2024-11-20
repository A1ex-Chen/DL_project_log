def get_asym_autoencoder_kl_config(block_out_channels=None, norm_num_groups
    =None):
    block_out_channels = block_out_channels or [2, 4]
    norm_num_groups = norm_num_groups or 2
    init_dict = {'in_channels': 3, 'out_channels': 3, 'down_block_types': [
        'DownEncoderBlock2D'] * len(block_out_channels),
        'down_block_out_channels': block_out_channels,
        'layers_per_down_block': 1, 'up_block_types': ['UpDecoderBlock2D'] *
        len(block_out_channels), 'up_block_out_channels':
        block_out_channels, 'layers_per_up_block': 1, 'act_fn': 'silu',
        'latent_channels': 4, 'norm_num_groups': norm_num_groups,
        'sample_size': 32, 'scaling_factor': 0.18215}
    return init_dict
