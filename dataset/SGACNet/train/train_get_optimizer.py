def get_optimizer(args, model):
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay, momentum=args.momentum,
            nesterov=True)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay, betas=(0.9, 0.999))
    else:
        raise NotImplementedError(
            'Currently only SGD and Adam as optimizers are supported. Got {}'
            .format(args.optimizer))
    print('Using {} as optimizer'.format(args.optimizer))
    return optimizer
