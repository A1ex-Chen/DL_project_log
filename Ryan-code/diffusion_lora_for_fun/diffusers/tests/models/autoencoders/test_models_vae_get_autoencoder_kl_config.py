def get_autoencoder_kl_config(block_out_channels=None, norm_num_groups=None):
    block_out_channels = block_out_channels or [2, 4]
    norm_num_groups = norm_num_groups or 2
    init_dict = {'block_out_channels': block_out_channels, 'in_channels': 3,
        'out_channels': 3, 'down_block_types': ['DownEncoderBlock2D'] * len
        (block_out_channels), 'up_block_types': ['UpDecoderBlock2D'] * len(
        block_out_channels), 'latent_channels': 4, 'norm_num_groups':
        norm_num_groups}
    return init_dict
