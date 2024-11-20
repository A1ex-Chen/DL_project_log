def train(args, model=None, data_splits=None):
    torch.manual_seed(args.seed)
    writer = SummaryWriter(comment=f'{args.model}')
    data_splits = get_dataloaders(data_root=args.data_root, dataset_name=
        args.dataset_name, batch_size=args.batch_size, num_workers=args.workers
        )
    train_loader, test_loader = data_splits['train'], data_splits['test']
    model = get_model(model_name=args.model, dataset_name=args.dataset_name,
        pretrained=False)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=
        args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=
        args.milestones, gamma=args.lr_gamma)
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, train_loader, optimizer, epoch, writer, dryrun=
            args.dryrun)
        if args.dryrun:
            break
        test(model, test_loader, epoch, writer, dryrun=args.dryrun)
        scheduler.step()
    if not args.dryrun:
        torch.save(model.state_dict(), f'{args.model}_cifar100.pt')
