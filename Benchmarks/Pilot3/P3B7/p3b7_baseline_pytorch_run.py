def run(args):
    args = candle.ArgumentStruct(**args)
    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    train_data, valid_data = get_synthetic_data(args)
    hparams = Hparams(kernel1=args.kernel1, kernel2=args.kernel2, kernel3=
        args.kernel3, embed_dim=args.embed_dim, n_filters=args.n_filters)
    train_loader = DataLoader(train_data, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size)
    model = MTCNN(TASKS, hparams).to(args.device)
    model = create_prune_masks(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
        eps=args.eps)
    train_epoch_loss = []
    valid_epoch_loss = []
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, args.device, epoch)
        valid_loss = evaluate(model, valid_loader, args.device)
        train_epoch_loss.append(train_loss)
        valid_epoch_loss.append(valid_loss)
    model = remove_prune_masks(model)
