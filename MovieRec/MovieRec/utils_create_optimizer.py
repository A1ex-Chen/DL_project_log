def create_optimizer(model, args):
    if args.optimizer == 'Adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args
            .weight_decay)
    return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.
        weight_decay, momentum=args.momentum)
