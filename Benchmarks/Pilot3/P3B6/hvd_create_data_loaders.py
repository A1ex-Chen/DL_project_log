def create_data_loaders(args):
    """Initialize data loaders

    Args:
        gParameters: parameters from candle

    Returns:
        train, valid, test data loaders
    """
    train, valid, test = load_data(args)
    train_sampler = DistributedSampler(train, num_replicas=hvd.size(), rank
        =hvd.rank(), shuffle=True)
    valid_sampler = DistributedSampler(valid, num_replicas=hvd.size(), rank
        =hvd.rank(), shuffle=False)
    test_sampler = DistributedSampler(test, num_replicas=hvd.size(), rank=
        hvd.rank(), shuffle=False)
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=
        train_sampler)
    valid_loader = DataLoader(valid, batch_size=args.batch_size, sampler=
        valid_sampler)
    test_loader = DataLoader(test, batch_size=args.batch_size, sampler=
        test_sampler)
    return train_loader, train_sampler, valid_loader, test_loader
