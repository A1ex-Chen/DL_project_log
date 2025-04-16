def get_encoder(cfg, device, dataset=None, c_dim=0):
    """ Returns an encoder instance.

    If input type if 'idx', the encoder consists of an embedding layer.

    Args:
        cfg (yaml config): yaml config object
        device (device): PyTorch device
        dataset (dataset): dataset
        c_dim (int): dimension of conditioned code c
    """
    encoder = cfg['model']['encoder']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    if encoder == 'idx':
        if cfg['model']['learn_embedding']:
            encoder = nn.Sequential(nn.Embedding(len(dataset), 128), nn.
                Linear(128, c_dim)).to(device)
        else:
            encoder = nn.Embedding(len(dataset), c_dim).to(device)
    elif encoder is not None:
        encoder = encoder_dict[encoder](c_dim=c_dim, dim=3, **encoder_kwargs
            ).to(device)
    else:
        encoder = None
    return encoder
