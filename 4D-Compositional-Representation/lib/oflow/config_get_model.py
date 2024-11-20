def get_model(cfg, device=None, dataset=None, **kwargs):
    """ Returns an OFlow model instance.

    Depending on the experimental setup, it consists of encoders,
    latent encoders, a velocity field, and a decoder instance.

    Args:
        cfg (yaml config): yaml config object
        device (device): PyTorch device
        c_dim (int): dimension of conditioned code c
        z_dim (int): dimension of latent code z
    """
    dim = cfg['data']['dim']
    z_dim = cfg['model']['z_dim']
    c_dim = cfg['model']['c_dim']
    input_type = cfg['data']['input_type']
    ode_solver = cfg['model']['ode_solver']
    ode_step_size = cfg['model']['ode_step_size']
    use_adjoint = cfg['model']['use_adjoint']
    rtol = cfg['model']['rtol']
    atol = cfg['model']['atol']
    decoder = get_decoder(cfg, device, dim, c_dim, z_dim)
    velocity_field = get_velocity_field(cfg, device, dim, c_dim, z_dim)
    encoder = get_encoder(cfg, device, dataset, c_dim)
    encoder_latent = get_encoder_latent(cfg, device, c_dim, z_dim)
    encoder_latent_temporal = get_encoder_latent_temporal(cfg, device,
        c_dim, z_dim)
    encoder_temporal = get_encoder_temporal(cfg, device, dataset, c_dim, z_dim)
    p0_z = get_prior_z(cfg, device)
    model = models.OccupancyFlow(decoder=decoder, encoder=encoder,
        encoder_latent=encoder_latent, encoder_latent_temporal=
        encoder_latent_temporal, encoder_temporal=encoder_temporal,
        vector_field=velocity_field, ode_step_size=ode_step_size,
        use_adjoint=use_adjoint, rtol=rtol, atol=atol, ode_solver=
        ode_solver, p0_z=p0_z, device=device, input_type=input_type)
    return model
