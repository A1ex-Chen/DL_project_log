def create_vae_diffusers_config(original_config, checkpoint, image_size: int):
    """
    Creates a VAE config for diffusers based on the config of the original MusicLDM model. Compared to the original
    Stable Diffusion conversion, this function passes a *learnt* VAE scaling factor to the diffusers VAE.
    """
    vae_params = original_config['model']['params']['first_stage_config'][
        'params']['ddconfig']
    _ = original_config['model']['params']['first_stage_config']['params'][
        'embed_dim']
    block_out_channels = [(vae_params['ch'] * mult) for mult in vae_params[
        'ch_mult']]
    down_block_types = ['DownEncoderBlock2D'] * len(block_out_channels)
    up_block_types = ['UpDecoderBlock2D'] * len(block_out_channels)
    scaling_factor = checkpoint['scale_factor'
        ] if 'scale_by_std' in original_config['model']['params'] else 0.18215
    config = {'sample_size': image_size, 'in_channels': vae_params[
        'in_channels'], 'out_channels': vae_params['out_ch'],
        'down_block_types': tuple(down_block_types), 'up_block_types':
        tuple(up_block_types), 'block_out_channels': tuple(
        block_out_channels), 'latent_channels': vae_params['z_channels'],
        'layers_per_block': vae_params['num_res_blocks'], 'scaling_factor':
        float(scaling_factor)}
    return config
