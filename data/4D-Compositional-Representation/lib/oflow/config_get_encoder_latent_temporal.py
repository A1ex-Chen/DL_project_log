def get_encoder_latent_temporal(cfg, device, c_dim=0, z_dim=0):
    """ Returns a latent encoder instance.

    Args:  
        cfg (yaml config): yaml config object
        device (device): PyTorch device
        c_dim (int): dimension of conditioned code c
        z_dim (int): dimension of latent code z
    """
    encoder_latent_temporal_kwargs = cfg['model'][
        'encoder_latent_temporal_kwargs']
    encoder_latent_temporal = cfg['model']['encoder_latent_temporal']
    if encoder_latent_temporal:
        encoder_latent_temporal = models.encoder_latent_dict[
            encoder_latent_temporal](z_dim=z_dim, c_dim=c_dim, **
            encoder_latent_temporal_kwargs).to(device)
    else:
        encoder_latent_temporal = None
    return encoder_latent_temporal
