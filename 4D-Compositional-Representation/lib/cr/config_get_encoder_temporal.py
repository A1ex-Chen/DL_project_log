def get_encoder_temporal(cfg, device, c_dim=0):
    """ Returns a temporal encoder instance.

    Args:
        cfg (yaml): yaml config
        device (device): Pytorch device
        c_dim (int): dimension of latent conditioned code c
        z_dim (int): dimension of latent code z
    """
    encoder_temporal = cfg['model']['encoder_temporal']
    encoder_temporal_kwargs = cfg['model']['encoder_temporal_kwargs']
    length_sequence = cfg['data']['length_sequence']
    if encoder_temporal:
        encoder_temporal = encoder_temporal_dict[encoder_temporal](c_dim=
            c_dim, dim=length_sequence * 3, **encoder_temporal_kwargs).to(
            device)
    else:
        encoder_temporal = None
    return encoder_temporal
