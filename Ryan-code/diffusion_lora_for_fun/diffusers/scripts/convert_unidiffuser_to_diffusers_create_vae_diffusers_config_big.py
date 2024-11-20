def create_vae_diffusers_config_big():
    vae_config = {'sample_size': 256, 'in_channels': 3, 'out_channels': 3,
        'down_block_types': ['DownEncoderBlock2D', 'DownEncoderBlock2D',
        'DownEncoderBlock2D', 'DownEncoderBlock2D'], 'up_block_types': [
        'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D',
        'UpDecoderBlock2D'], 'block_out_channels': [128, 256, 512, 512],
        'latent_channels': 4, 'layers_per_block': 2}
    return vae_config
