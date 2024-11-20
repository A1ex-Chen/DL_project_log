def train_vq_vae_with_gated_pixelcnn_prior(args, train_set, validation_set,
    test_set):
    """
    args: parameters for code
    train_set
    validation_set
    test_set

    Returns
    - a (# of training iterations,) numpy array of VQ-VAE train losess evaluated every minibatch
    - a (# of epochs + 1,) numpy array of VQ-VAE train losses evaluated once at initialization and after each epoch
    - a (# of training iterations,) numpy array of PixelCNN prior train losess evaluated every minibatch
    - a (# of epochs + 1,) numpy array of PixelCNN prior train losses evaluated once at initialization and after each epoch
    - a (100, 32, 32, 3) numpy array of 100 samples (an equal number from each class) with values in {0, ... 255}
    - a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
        FROM THE TEST SET with values in [0, 255]
    """
    start_time = time.time()
    if args.dataset == 'mnist' or args.dataset == 'fashion-mnist':
        N, H, W = train_set.dataset.data.shape
        C = 1
    elif args.dataset == 'stl':
        N, C, H, W = train_set.dataset.data.shape
    else:
        N, H, W, C = train_set.dataset.data.shape
    data_shape = {'N': N, 'H': H, 'W': W, 'C': C}
    batch_size = args.batch_size
    dataset_params = {'batch_size': batch_size, 'shuffle': False}
    print('[INFO] Creating model and data loaders')
    train_loader = torch.utils.data.DataLoader(train_set, **dataset_params)
    validation_loader = torch.utils.data.DataLoader(validation_set, **
        dataset_params)
    test_loader = torch.utils.data.DataLoader(test_set, **dataset_params)
    n_epochs_vae = args.epochs
    n_epochs_cnn = args.epochs
    lr = args.lr
    K = args.num_embeddings
    D = args.embedding_dim
    vq_vae = VQVAE(K=K, D=D, channel=C)
    if args.dp == 'gaussian':
        vq_vae = opacus.validators.ModuleValidator.fix(vq_vae)
        """
        if args.clip_per_layer:
            # Each layer has the same clipping threshold. The total grad norm is still bounded by `args.max_per_sample_grad_norm`.
            n_layers = len(
                [(n, p) for n, p in vq_vae.named_parameters() if p.requires_grad]
            )
            max_grad_norm = [
                                args.max_per_sample_grad_norm / np.sqrt(n_layers)
                            ] * n_layers
        else:
            max_grad_norm = args.max_per_sample_grad_norm
        """
        privacy_engine = PrivacyEngine()
        vq_vae, optimizer, train_loader = privacy_engine.make_private(module
            =vq_vae, optimizer=torch.optim.Adam(vq_vae.parameters(), lr=
            args.lr, weight_decay=args.weight_decay, betas=(args.beta1,
            args.beta2)), data_loader=train_loader, noise_multiplier=args.
            sigma, max_grad_norm=args.max_per_sample_grad_norm)
    else:
        optimizer = torch.optim.Adam(vq_vae.parameters(), lr=lr)
    vq_vae_metrics = train(args, data_shape, vq_vae, optimizer,
        n_epochs_vae, train_loader, validation_loader, test_loader,
        privacy_engine)
    print(f'[DONE] Time elapsed: {time.time() - start_time:.2f} s')
    return vq_vae_metrics
