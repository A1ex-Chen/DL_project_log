def get_prior_z(cfg, device, **kwargs):
    """ Returns prior distribution.

    Args:
        cfg (yaml): yaml config
        device (device): Pytorch device
    """
    z_dim = cfg['model']['z_dim']
    p0_z = dist.Normal(torch.zeros(z_dim, device=device), torch.ones(z_dim,
        device=device))
    return p0_z
