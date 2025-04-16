def get_encoder(model_path: str, dataframe: pd.DataFrame, autoencoder_init:
    bool, layer_dim: int, num_layers: int, latent_dim: int, ae_loss_func:
    str, ae_opt: str, ae_lr: float, ae_reg: float, lr_decay_factor: float,
    max_num_epochs: int, early_stop_patience: int, validation_ratio: float=
    0.2, trn_batch_size: int=32, val_batch_size: int=1024, device: torch.
    device=torch.device('cuda'), verbose: bool=True, rand_state: int=0):
    """encoder = gene_encoder = get_encoder(./models/', dataframe,
           True, 1000, 3, 100, 'mse', 'sgd', 1e-3, 0.98, 100, 10)

    This function constructs, initializes and returns a feature encoder for
    the given dataframe.

    When parameter autoencoder_init is set to False, it simply construct and
    return an encoder with simple initialization (nn.init.xavier_normal_).

    When autoencoder_init is set to True, the function will return the
    encoder part of an autoencoder trained on the given data. It will first
    check if the model file exists. If not, it will start training with
    given training hyper-parameters.

    Note that the saved model in disk contains the whole autoencoder (
    encoder and decoder). But the function only returns the encoder.

    Also the dataframe should be processed before this function call.

    Args:
        model_path (str): path to model for loading (if exists) and
            saving (for future usage).
        dataframe (pd.DataFrame): dataframe for training and validation.

        autoencoder_init (bool): indicator for using autoencoder as feature
            encoder initialization method. If True, the function will
            construct a autoencoder with symmetric encoder and decoder,
            and then train it with part of dataframe while validating on
            the rest, until early stopping is evoked or running out of epochs.
        layer_dim (int): layer dimension for feature encoder.
        num_layers (int): number of layers for feature encoder.
        latent_dim (int): latent (output) space dimension for feature encoder.

        ae_loss_func (str): loss function for autoencoder training. Select
            between 'mse' and 'l1'.
        ae_opt (str): optimizer for autoencoder training. Select between
            'SGD', 'Adam', and 'RMSprop'.
        ae_lr (float): learning rate for autoencoder training.
        lr_decay_factor (float): exponential learning rate decay factor.
        max_num_epochs (int): maximum number of epochs allowed.
        early_stop_patience (int): patience for early stopping. If the
            validation loss does not increase for this many epochs, the
            function returns the encoder part of the autoencoder, with the
            best validation loss so far.

        validation_ratio (float): (validation data size / overall data size).
        trn_batch_size (int): batch size for training.
        val_batch_size (int): batch size for validation.

        device (torch.device): torch device indicating where to train:
            either on CPU or GPU. Note that this function does not support
            multi-GPU yet.
        verbose (bool): indicator for training epoch log on terminal.
        rand_state (int): random seed used for layer initialization,
            training/validation splitting, and all other processes that
            requires randomness.

    Returns:
        torch.nn.Module: encoder for features from given dataframe.
    """
    if not autoencoder_init:
        return EncNet(input_dim=dataframe.shape[1], layer_dim=layer_dim,
            num_layers=num_layers, latent_dim=latent_dim, autoencoder=False
            ).to(device).encoder
    if os.path.exists(model_path):
        logger.debug('Loading existing autoencoder model from %s ...' %
            model_path)
        model = EncNet(input_dim=dataframe.shape[1], layer_dim=layer_dim,
            latent_dim=latent_dim, num_layers=num_layers, autoencoder=True).to(
            device)
        model.load_state_dict(torch.load(model_path))
        return model.encoder
    logger.debug('Constructing autoencoder from dataframe ...')
    seed_random_state(rand_state)
    trn_df, val_df = train_test_split(dataframe, test_size=validation_ratio,
        random_state=rand_state, shuffle=True)
    dataloader_kwargs = {'shuffle': 'True', 'num_workers': 4 if device ==
        torch.device('cuda') else 0, 'pin_memory': True if device == torch.
        device('cuda') else False}
    trn_dataloader = DataLoader(DataFrameDataset(trn_df), batch_size=
        trn_batch_size, **dataloader_kwargs)
    val_dataloader = DataLoader(DataFrameDataset(val_df), batch_size=
        val_batch_size, **dataloader_kwargs)
    autoencoder = EncNet(input_dim=dataframe.shape[1], layer_dim=layer_dim,
        latent_dim=latent_dim, num_layers=num_layers, autoencoder=True).to(
        device)
    assert ae_loss_func.lower() == 'l1' or ae_loss_func.lower() == 'mse'
    loss_func = F.l1_loss if ae_loss_func.lower() == 'l1' else F.mse_loss
    optimizer = get_optimizer(opt_type=ae_opt, networks=autoencoder,
        learning_rate=ae_lr, l2_regularization=ae_reg)
    lr_decay = LambdaLR(optimizer, lr_lambda=lambda e: lr_decay_factor ** e)
    best_val_loss = np.inf
    best_autoencoder = None
    patience = 0
    if verbose:
        print('=' * 80)
        print('Training log for autoencoder model (%s): ' % model_path)
    for epoch in range(max_num_epochs):
        lr_decay.step(epoch)
        autoencoder.train()
        trn_loss = 0.0
        for batch_idx, samples in enumerate(trn_dataloader):
            samples = samples.to(device)
            recon_samples = autoencoder(samples)
            autoencoder.zero_grad()
            loss = loss_func(input=recon_samples, target=samples)
            loss.backward()
            optimizer.step()
            trn_loss += loss.item() * len(samples)
        trn_loss /= len(trn_dataloader.dataset)
        autoencoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for samples in val_dataloader:
                samples = samples.to(device)
                recon_samples = autoencoder(samples)
                loss = loss_func(input=recon_samples, target=samples)
                val_loss += loss.item() * len(samples)
            val_loss /= len(val_dataloader.dataset)
        if verbose:
            print('Epoch %4i: training loss: %.4f;\t validation loss: %.4f' %
                (epoch + 1, trn_loss, val_loss))
        if val_loss < best_val_loss:
            patience = 0
            best_val_loss = val_loss
            best_autoencoder = copy.deepcopy(autoencoder)
        else:
            patience += 1
            if patience > early_stop_patience:
                if verbose:
                    print(
                        'Evoking early stopping. Best validation loss %.4f.' %
                        best_val_loss)
                break
    try:
        os.makedirs(os.path.dirname(model_path))
    except FileExistsError:
        pass
    torch.save(best_autoencoder.state_dict(), model_path)
    return best_autoencoder.encoder
