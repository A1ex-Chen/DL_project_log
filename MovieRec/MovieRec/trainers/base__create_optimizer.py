def _create_optimizer(self):
    args = self.args
    if args.optimizer.lower() == 'adam':
        return optim.Adam(self.model.parameters(), lr=args.lr, weight_decay
            =args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=
            args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError
