def get_model(cfg, device=None, dataset=None, **kwargs):
    """ Returns a model instance.

    Args:
        cfg (yaml): yaml config
        device (device): Pytorch device
        dataset (dataset): Pytorch dataset
    """
    dim = cfg['data']['dim']
    c_dim = cfg['model']['c_dim']
    input_type = cfg['data']['input_type']
    ode_solver = cfg['model']['ode_solver']
    ode_step_size = cfg['model']['ode_step_size']
    use_adjoint = cfg['model']['use_adjoint']
    rtol = cfg['model']['rtol']
    atol = cfg['model']['atol']
    decoder = get_decoder(cfg, device, c_dim)
    encoder = get_encoder(cfg, device, dataset, c_dim=128)
    encoder_motion = get_encoder_temporal(cfg, device, c_dim)
    encoder_identity = get_encoder(cfg, device, dataset, c_dim=128)
    velocity_field = get_velocity_field(cfg, device, dim, c_dim)
    model = models.Compositional4D(decoder=decoder, encoder=encoder,
        encoder_motion=encoder_motion, encoder_identity=encoder_identity,
        vector_field=velocity_field, ode_step_size=ode_step_size,
        use_adjoint=use_adjoint, rtol=rtol, atol=atol, ode_solver=
        ode_solver, device=device, input_type=input_type)
    return model
