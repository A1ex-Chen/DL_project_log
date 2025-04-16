def run(params):
    args = candle.ArgumentStruct(**params)
    args.cuda = torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    darts.banner(device=device)
    train_data = darts.Uno('./data', 'train', download=True)
    valid_data = darts.Uno('./data', 'test')
    trainloader = DataLoader(train_data, batch_size=args.batch_size)
    validloader = DataLoader(valid_data, batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss().to(device)
    tasks = {'response': 2}
    model = darts.LinearNetwork(input_dim=942, tasks=tasks, criterion=
        criterion, device=device).to(device)
    architecture = darts.Architecture(model, args, device=device)
    optimizer = optim.SGD(model.parameters(), args.learning_rate, momentum=
        args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.
        epochs), eta_min=args.learning_rate_min)
    train_meter = darts.EpochMeter(tasks, 'train')
    valid_meter = darts.EpochMeter(tasks, 'valid')
    genotype_store = darts.GenotypeStorage(root=args.save_path)
    for epoch in range(args.epochs):
        lr = scheduler.get_lr()[0]
        logger.info(f'\nEpoch: {epoch} lr: {lr}')
        genotype = model.genotype()
        logger.info(f'Genotype: {genotype}\n')
        train(trainloader, model, architecture, criterion, optimizer,
            scheduler, args, tasks, train_meter, genotype, genotype_store,
            device)
        validate(validloader, model, criterion, args, tasks, valid_meter,
            device)
