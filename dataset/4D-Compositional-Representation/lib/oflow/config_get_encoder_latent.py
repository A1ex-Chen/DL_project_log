def get_encoder_latent(cfg, device, c_dim=0, z_dim=0):
    """ Returns a latent encoder instance.

    Args:
        cfg (yaml config): yaml config object
        device (device): PyTorch device
        c_dim (int): dimension of conditioned code c
        z_dim (int): dimension of latent code z
    """
    encoder_latent_kwargs = cfg['model']['encoder_latent_kwargs']
    encoder_latent = cfg['model']['encoder_latent']
    if encoder_latent:
        encoder_latent = models.encoder_latent_dict[encoder_latent](z_dim=
            z_dim, c_dim=c_dim, **encoder_latent_kwargs).to(device)
    else:
        encoder_latent = None
    return encoder_latent
