def run(params):
    args = candle.ArgumentStruct(**params)
    args.cuda = torch.cuda.is_available()
    device = torch.device(f'cuda' if args.cuda else 'cpu')
    darts.banner(device=device)
    datapath = fetch_data(params)
    train_data = darts.P3B3(datapath, 'train')
    valid_data = darts.P3B3(datapath, 'test')
    trainloader = DataLoader(train_data, batch_size=args.batch_size)
    validloader = DataLoader(valid_data, batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss().to(device)
    tasks = {'subsite': 15, 'laterality': 3, 'behavior': 3, 'grade': 3}
    train_meter = darts.EpochMeter(tasks, 'train')
    valid_meter = darts.EpochMeter(tasks, 'valid')
    model = darts.ConvNetwork(tasks=tasks, criterion=criterion, device=device
        ).to(device)
    architecture = darts.Architecture(model, args, device=device)
    optimizer = optim.SGD(model.parameters(), args.learning_rate, momentum=
        args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.
        epochs), eta_min=args.learning_rate_min)
    genotype_store = darts.GenotypeStorage(root=args.save_path)
    min_loss = 9999
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        print(f'\nEpoch: {epoch} lr: {lr}')
        genotype = model.genotype()
        print(f'Genotype: {genotype}')
        train(trainloader, validloader, model, architecture, criterion,
            optimizer, lr, args, tasks, device, train_meter)
        valid_loss = infer(validloader, model, criterion, args, tasks,
            device, valid_meter)
        if valid_loss < min_loss:
            genotype_store.save_genotype(genotype)
            min_loss = valid_loss
