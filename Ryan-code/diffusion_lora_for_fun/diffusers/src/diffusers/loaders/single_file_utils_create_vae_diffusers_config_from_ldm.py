def create_vae_diffusers_config_from_ldm(original_config, checkpoint,
    image_size=None, scaling_factor=None):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    if image_size is not None:
        deprecation_message = (
            'Configuring AutoencoderKL with the `image_size` argumentis deprecated and will be ignored in future versions.'
            )
        deprecate('image_size', '1.0.0', deprecation_message)
    image_size = set_image_size(checkpoint, image_size=image_size)
    if 'edm_mean' in checkpoint and 'edm_std' in checkpoint:
        latents_mean = checkpoint['edm_mean']
        latents_std = checkpoint['edm_std']
    else:
        latents_mean = None
        latents_std = None
    vae_params = original_config['model']['params']['first_stage_config'][
        'params']['ddconfig']
    if (scaling_factor is None and latents_mean is not None and latents_std
         is not None):
        scaling_factor = PLAYGROUND_VAE_SCALING_FACTOR
    elif scaling_factor is None and 'scale_factor' in original_config['model'][
        'params']:
        scaling_factor = original_config['model']['params']['scale_factor']
    elif scaling_factor is None:
        scaling_factor = LDM_VAE_DEFAULT_SCALING_FACTOR
    block_out_channels = [(vae_params['ch'] * mult) for mult in vae_params[
        'ch_mult']]
    down_block_types = ['DownEncoderBlock2D'] * len(block_out_channels)
    up_block_types = ['UpDecoderBlock2D'] * len(block_out_channels)
    config = {'sample_size': image_size, 'in_channels': vae_params[
        'in_channels'], 'out_channels': vae_params['out_ch'],
        'down_block_types': down_block_types, 'up_block_types':
        up_block_types, 'block_out_channels': block_out_channels,
        'latent_channels': vae_params['z_channels'], 'layers_per_block':
        vae_params['num_res_blocks'], 'scaling_factor': scaling_factor}
    if latents_mean is not None and latents_std is not None:
        config.update({'latents_mean': latents_mean, 'latents_std':
            latents_std})
    return config
