def run(params):
    args = candle.ArgumentStruct(**params)
    args.cuda = torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    darts.banner(device=device)
    trainloader = torch.utils.data.DataLoader(datasets.MNIST('./data',
        train=True, download=True, transform=transforms.Compose([transforms
        .ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=args.batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(datasets.MNIST('./data',
        train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        ), batch_size=args.batch_size, shuffle=True)
    tasks = {'digits': 10}
    criterion = nn.CrossEntropyLoss().to(device)
    stem = Stem(cell_dim=100)
    model = darts.Network(stem, cell_dim=100, classifier_dim=676, ops=OPS,
        tasks=tasks, criterion=criterion, device=device).to(device)
    architecture = darts.Architecture(model, args, device=device)
    optimizer = optim.SGD(model.parameters(), args.learning_rate, momentum=
        args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.
        epochs), eta_min=args.learning_rate_min)
    train_meter = darts.EpochMeter(tasks, 'train')
    valid_meter = darts.EpochMeter(tasks, 'valid')
    for epoch in range(args.epochs):
        lr = scheduler.get_lr()[0]
        logger.info(f'\nEpoch: {epoch} lr: {lr}')
        genotype = model.genotype()
        logger.info(f'Genotype: {genotype}\n')
        train(trainloader, model, architecture, criterion, optimizer,
            scheduler, args, tasks, train_meter, device)
        validate(validloader, model, criterion, args, tasks, valid_meter,
            device)
