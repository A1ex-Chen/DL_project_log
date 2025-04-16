def get_prior_z(cfg, device, **kwargs):
    """ Returns the prior distribution of latent code z.

    Args:
        cfg (yaml config): yaml config object
        device (device): PyTorch device
    """
    z_dim = cfg['model']['z_dim']
    p0_z = dist.Normal(torch.zeros(z_dim, device=device), torch.ones(z_dim,
        device=device))
    return p0_z
