def get_gene_encoder(model_folder: str, data_root: str,
    rnaseq_feature_usage: str, rnaseq_scaling: str, autoencoder_init: bool,
    layer_dim: int, num_layers: int, latent_dim: int, training_kwarg: dict,
    device: torch.device=torch.device('cuda'), verbose: bool=True,
    rand_state: int=0):
    """gene_encoder = get_gene_encoder(./models/', './data/',
           'source_scale', 'std', True, 1000, 3, 100, training_kwarg_dict)

    This function takes arguments about RNA sequence encoder and return the
    corresponding encoders. It will execute one of the following based on
    parameters and previous saved models:
        * simply initialize a new encoder;
        * load existing autoencoder and return the encoder part;
        * train a new autoencoder and return the encoder part;

    Note that this function requires existing dataframes of RNA sequence.

    Args:
        model_folder (str): path to the model folder.
        data_root (str): path to data folder (root).

        rnaseq_feature_usage (str): RNA sequence data used. Choose between
            'source_scale' and 'combat'.
        rnaseq_scaling (str): Scaling method for RNA sequence data.

        autoencoder_init (bool): indicator for using autoencoder as RNA
            sequence encoder initialization method.
        layer_dim (int): layer dimension for RNA sequence encoder.
        num_layers (int): number of layers for RNA sequence encoder.
        latent_dim (int): latent (output) space dimension for RNA sequence
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
        torch.nn.Module: encoder for RNA sequence dataframe.
    """
    gene_encoder_name = 'gene_net(%i*%i=>%i, %s, scaling=%s).pt' % (layer_dim,
        num_layers, latent_dim, rnaseq_feature_usage, rnaseq_scaling)
    gene_encoder_path = os.path.join(model_folder, gene_encoder_name)
    rna_seq_df = get_rna_seq_df(data_root=data_root, rnaseq_feature_usage=
        rnaseq_feature_usage, rnaseq_scaling=rnaseq_scaling)
    rna_seq_df.drop_duplicates(inplace=True)
    return get_encoder(model_path=gene_encoder_path, dataframe=rna_seq_df,
        autoencoder_init=autoencoder_init, layer_dim=layer_dim, num_layers=
        num_layers, latent_dim=latent_dim, **training_kwarg, device=device,
        verbose=verbose, rand_state=rand_state)
