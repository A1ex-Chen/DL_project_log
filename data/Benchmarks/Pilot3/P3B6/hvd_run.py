def run(args):
    train_loader, train_sampler, valid_loader, test_loader = (
        create_data_loaders(args))
    model = model = HiBERT(args.pretrained_weights_path, args.num_classes)
    model.to(args.device)
    params = [{'params': [p for n, p in model.named_parameters()],
        'weight_decay': args.weight_decay}]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, eps=args.eps)
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.
        named_parameters())
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(args.num_epochs):
        train(train_loader, train_sampler, model, optimizer, criterion,
            args, epoch)
        validate(valid_loader, model, args, epoch)
