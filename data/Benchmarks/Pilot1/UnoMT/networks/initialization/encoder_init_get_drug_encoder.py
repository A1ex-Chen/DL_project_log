def get_drug_encoder(model_folder: str, data_root: str, drug_feature_usage:
    str, dscptr_scaling: str, dscptr_nan_threshold: float, autoencoder_init:
    bool, layer_dim: int, num_layers: int, latent_dim: int, training_kwarg:
    dict, device: torch.device=torch.device('cuda'), verbose: bool=True,
    rand_state: int=0):
    """drug_encoder = get_gene_encoder(./models/', './data/',
               'both', 'std', 0.,  True, 1000, 3, 100, training_kwarg_dict)

    This function takes arguments about drug feature encoder and return the
    corresponding encoders. It will execute one of the following based on
    parameters and previous saved models:
        * simply initialize a new encoder;
        * load existing autoencoder and return the encoder part;
        * train a new autoencoder and return the encoder part;

    Note that this function requires existing dataframes of drug feature.


    Args:
        model_folder (str): path to the model folder.
        data_root (str): path to data folder (root).

        drug_feature_usage (str): Drug feature usage used. Choose between
            'fingerprint', 'descriptor', or 'both'.
        dscptr_scaling (str): Scaling method for drug feature data.
        dscptr_nan_threshold (float): ratio of NaN values allowed for drug
            features. Unqualified columns and rows will be dropped.

        autoencoder_init (bool): indicator for using autoencoder as drug
            feature encoder initialization method.
        layer_dim (int): layer dimension for drug feature encoder.
        num_layers (int): number of layers for drug feature encoder.
        latent_dim (int): latent (output) space dimension for drug feature
            encoder.

        training_kwarg (dict): training parameters in dict format,
            which contains all the training parameters in get_encoder
            function. Please refer to get_encoder for more details.

        device (torch.device): torch device indicating where to train:
            either on CPU or GPU. Note that this function does not support
            multi-GPU yet.
        verbose (bool): indicator for training epoch log on terminal.
        rand_state (int): random seed used for layer initialization,
            training/validation splitting, and all other processes that
            requires randomness.

    Returns:
        torch.nn.Module: encoder for drug feature dataframe.
    """
    drug_encoder_name = (
        'drug_net(%i*%i=>%i, %s, descriptor_scaling=%s, nan_thresh=%.2f).pt' %
        (layer_dim, num_layers, latent_dim, drug_feature_usage,
        dscptr_scaling, dscptr_nan_threshold))
    drug_encoder_path = os.path.join(model_folder, drug_encoder_name)
    drug_feature_df = get_drug_feature_df(data_root=data_root,
        drug_feature_usage=drug_feature_usage, dscptr_scaling=
        dscptr_scaling, dscptr_nan_thresh=dscptr_nan_threshold)
    return get_encoder(model_path=drug_encoder_path, dataframe=
        drug_feature_df, autoencoder_init=autoencoder_init, layer_dim=
        layer_dim, num_layers=num_layers, latent_dim=latent_dim, **
        training_kwarg, device=device, verbose=verbose, rand_state=rand_state)
