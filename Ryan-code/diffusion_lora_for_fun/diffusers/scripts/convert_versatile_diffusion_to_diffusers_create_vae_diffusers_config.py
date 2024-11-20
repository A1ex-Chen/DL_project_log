def create_vae_diffusers_config(vae_params):
    """
    Creates a config for the diffusers based on the config of the VD model.
    """
    block_out_channels = [(vae_params.ch * mult) for mult in vae_params.ch_mult
        ]
    down_block_types = ['DownEncoderBlock2D'] * len(block_out_channels)
    up_block_types = ['UpDecoderBlock2D'] * len(block_out_channels)
    config = {'sample_size': vae_params.resolution, 'in_channels':
        vae_params.in_channels, 'out_channels': vae_params.out_ch,
        'down_block_types': tuple(down_block_types), 'up_block_types':
        tuple(up_block_types), 'block_out_channels': tuple(
        block_out_channels), 'latent_channels': vae_params.z_channels,
        'layers_per_block': vae_params.num_res_blocks}
    return config
