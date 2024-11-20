def create_vae_diffusers_config_test():
    vae_config = {'sample_size': 32, 'in_channels': 3, 'out_channels': 3,
        'down_block_types': ['DownEncoderBlock2D', 'DownEncoderBlock2D'],
        'up_block_types': ['UpDecoderBlock2D', 'UpDecoderBlock2D'],
        'block_out_channels': [32, 64], 'latent_channels': 4,
        'layers_per_block': 1}
    return vae_config
