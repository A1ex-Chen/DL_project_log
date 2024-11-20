def get_decoder(cfg, device, dim=3, c_dim=0, z_dim=0):
    """ Returns a decoder instance.

    Args:  
        cfg (yaml config): yaml config object
        device (device): PyTorch device
        dim (int): points dimension
        c_dim (int): dimension of conditioned code c
        z_dim (int): dimension of latent code z
    """
    decoder = cfg['model']['decoder']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    if decoder:
        decoder = models.decoder_dict[decoder](dim=dim, z_dim=z_dim, c_dim=
            c_dim, **decoder_kwargs).to(device)
    else:
        decoder = None
    return decoder
