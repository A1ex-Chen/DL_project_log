def get_velocity_field(cfg, device, dim=3, c_dim=0, z_dim=0):
    """ Returns a velocity field instance.

    Args:  
        cfg (yaml config): yaml config object
        device (device): PyTorch device
        dim (int): points dimension
        c_dim (int): dimension of conditioned code c
        z_dim (int): dimension of latent code z
    """
    velocity_field = cfg['model']['velocity_field']
    velocity_field_kwargs = cfg['model']['velocity_field_kwargs']
    if velocity_field:
        velocity_field = models.velocity_field_dict[velocity_field](out_dim
            =dim, z_dim=z_dim, c_dim=c_dim, **velocity_field_kwargs).to(device)
    else:
        velocity_field = None
    return velocity_field
