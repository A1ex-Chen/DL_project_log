def deepview_model_provider():
    args = get_args()
    vocab_size = 32317
    model_config = {'hidden_size': args.hidden_size, 'num_layers': args.
        num_layers, 'dropout': args.dropout, 'batch_first': False,
        'share_embedding': args.share_embedding}
    model = GNMTWithLoss(GNMT(vocab_size=vocab_size, **model_config),
        build_criterion(vocab_size, config.PAD, args.smoothing)).cuda()
    model.zero_grad()
    return model
