def get_encoder_temporal(cfg, device, dataset=None, c_dim=0, z_dim=0):
    """ Returns a temporal encoder instance.

    Args:  
        cfg (yaml config): yaml config object
        device (device): PyTorch device
        c_dim (int): dimension of conditioned code c
        z_dim (int): dimension of latent code z
    """
    encoder_temporal = cfg['model']['encoder_temporal']
    encoder_temporal_kwargs = cfg['model']['encoder_temporal_kwargs']
    if encoder_temporal:
        if encoder_temporal == 'idx':
            if cfg['model']['learn_embedding']:
                encoder_temporal = nn.Sequential(nn.Embedding(len(dataset),
                    128), nn.Linear(128, c_dim)).to(device)
            else:
                encoder_temporal = nn.Embedding(len(dataset), c_dim).to(device)
        else:
            encoder_temporal = encoder_temporal_dict[encoder_temporal](c_dim
                =c_dim, **encoder_temporal_kwargs).to(device)
    else:
        encoder_temporal = None
    return encoder_temporal
