def get_decoder(cfg, device, c_dim=0):
    """ Returns a decoder instance.

    Args:
        cfg (yaml config): yaml config
        device (device): Pytorch device
        c_dim (int): dimension of latent conditioned code c
        z_dim (int): dimension of latent code z
    """
    decoder = cfg['model']['decoder']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    inp_dim = c_dim * 2
    if decoder:
        decoder = models.decoder_dict[decoder](c_dim=inp_dim, **decoder_kwargs
            ).to(device)
    else:
        decoder = None
    return decoder
